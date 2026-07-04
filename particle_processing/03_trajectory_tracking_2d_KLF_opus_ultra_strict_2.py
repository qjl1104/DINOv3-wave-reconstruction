# particle_processing/03_trajectory_tracking_2d_strict.py
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
import os
import cv2
import glob
from collections import deque


class StrictWaveKalmanFilter:
    """严格的卡尔曼滤波器，防止轨迹跳跃"""

    def __init__(self, initial_pos):
        # 状态向量: [x, y, vx, vy]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)

        # 较小的初始不确定性，以减少轨迹跳跃
        self.P = np.eye(4)
        self.P[0:2, 0:2] *= 5.0
        self.P[2:4, 2:4] *= 10.0

        # 适中的过程噪声，允许一定的运动变化
        self.Q = np.eye(4)
        self.Q[0:2, 0:2] *= 3.0
        self.Q[2:4, 2:4] *= 10.0

        # 较小的测量噪声，更相信检测结果
        self.R = np.eye(2) * 5.0

        self.dt = 1.0

    def predict(self):
        """预测步骤，包含轻微的速度衰减"""
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 0.9, 0],  # 速度衰减
            [0, 0, 0, 0.9]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        return self.x[:2]

    def update(self, m):
        """更新步骤"""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        mv = np.array(m).reshape(2, 1)
        xv = self.x.reshape(4, 1)

        y = mv - H @ xv
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = (xv + K @ y).flatten()
        self.P = (np.eye(4) - K @ H) @ self.P


class StrictTrack:
    """严格的轨迹类"""

    def __init__(self, track_id, initial_detection, frame_idx):
        self.id = track_id
        self.kf = StrictWaveKalmanFilter(initial_detection)
        self.points = {frame_idx: initial_detection}

        # 状态
        self.age = 0
        self.consecutive_misses = 0
        self.total_visible_count = 1
        self.last_frame_seen = frame_idx
        self.birth_frame = frame_idx

        # 质量控制
        self.match_distances = deque(maxlen=20)
        self.positions = deque(maxlen=30)
        self.positions.append(initial_detection)

        # 运动一致性
        self.motion_consistency = 1.0
        self.max_allowed_distance = 40  # 最大允许匹配距离

    def predict(self):
        return self.kf.predict()

    def update(self, detection, frame_idx, match_distance):
        # 严格的距离检查
        if self.match_distances:
            avg_dist = np.mean(self.match_distances)
            # 如果距离突然增大，拒绝更新
            if match_distance > min(avg_dist * 2, self.max_allowed_distance):
                return False

        # 检查运动一致性
        if len(self.positions) >= 3:
            # 计算预期位置（基于最近的运动趋势）
            recent_positions = list(self.positions)[-3:]
            velocity = np.array(recent_positions[-1]) - np.array(recent_positions[-2])
            expected_pos = np.array(recent_positions[-1]) + velocity

            # 如果检测位置偏离预期太多，拒绝
            deviation = np.linalg.norm(np.array(detection) - expected_pos)
            if deviation > 50:
                self.motion_consistency *= 0.8
                if self.motion_consistency < 0.5:
                    return False
            else:
                self.motion_consistency = min(1.0, self.motion_consistency * 1.05)

        # 更新
        self.kf.update(np.array(detection))
        self.points[frame_idx] = detection
        self.match_distances.append(match_distance)
        self.positions.append(detection)

        self.age = 0
        self.consecutive_misses = 0
        self.total_visible_count += 1
        self.last_frame_seen = frame_idx

        return True

    def mark_missed(self):
        self.age += 1
        self.consecutive_misses += 1
        self.motion_consistency *= 0.95

    def get_ordered_points(self):
        return [self.points[fi] for fi in sorted(self.points.keys())]

    def get_density(self):
        if self.last_frame_seen >= self.birth_frame:
            lifetime = self.last_frame_seen - self.birth_frame + 1
            return self.total_visible_count / lifetime if lifetime > 0 else 0
        return 0

    def get_average_match_distance(self):
        return np.mean(self.match_distances) if self.match_distances else float('inf')

    def is_high_quality(self):
        """严格的质量判断"""
        return (len(self.points) >= 30 and
                self.get_density() > 0.7 and
                self.get_average_match_distance() < 25 and
                self.motion_consistency > 0.6)

    def should_terminate(self, max_age):
        return (self.consecutive_misses > max_age or
                self.motion_consistency < 0.3 or
                (self.consecutive_misses > max_age // 2 and self.get_density() < 0.5))


def strict_tracking(all_detections_per_frame, params, preprocessed_images_list=None, show_live_video=False):
    """严格的跟踪算法"""
    max_age = params['max_age']
    min_hits = params['min_hits_to_confirm']
    dist_thresh = params['dist_thresh']
    max_tracks = params.get('max_concurrent_tracks', 500)

    active_tracks = []
    completed_tracks = []
    next_track_id = 0

    stats = {
        'created': 0,
        'completed': 0,
        'rejected_matches': 0,
        'max_concurrent': 0
    }

    for frame_idx, detections in enumerate(all_detections_per_frame):
        # 预测
        predictions = []
        for track in active_tracks:
            predictions.append(track.predict())

        # 匹配
        matched_tracks = set()
        matched_detections = set()

        if active_tracks and detections:
            # 成本矩阵
            cost_matrix = np.full((len(active_tracks), len(detections)), 1000.0)

            for t_idx, (track, pred) in enumerate(zip(active_tracks, predictions)):
                for d_idx, det in enumerate(detections):
                    dist = np.linalg.norm(pred - np.array(det))

                    # 严格的距离限制
                    if dist < min(dist_thresh, track.max_allowed_distance):
                        cost_matrix[t_idx, d_idx] = dist

            # 匈牙利算法
            if np.any(cost_matrix < 1000):
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    for r, c in zip(row_ind, col_ind):
                        if cost_matrix[r, c] < min(dist_thresh, active_tracks[r].max_allowed_distance):
                            # 尝试更新
                            success = active_tracks[r].update(detections[c], frame_idx, cost_matrix[r, c])
                            if success:
                                matched_tracks.add(r)
                                matched_detections.add(c)
                            else:
                                stats['rejected_matches'] += 1
                except:
                    pass

        # 处理未匹配的轨迹
        new_active = []
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.mark_missed()

                if track.should_terminate(max_age):
                    if len(track.points) >= min_hits:
                        completed_tracks.append(track)
                        stats['completed'] += 1
                else:
                    new_active.append(track)
            else:
                new_active.append(track)

        active_tracks = new_active

        # 创建新轨迹（严格限制）
        if len(active_tracks) < max_tracks:
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_detections:
                    # 检查是否离现有轨迹太近
                    too_close = False
                    for track in active_tracks:
                        if track.last_frame_seen >= frame_idx - 2:
                            last_pos = track.points.get(track.last_frame_seen)
                            if last_pos and np.linalg.norm(np.array(det) - np.array(last_pos)) < 50:
                                too_close = True
                                break

                    if not too_close and len(active_tracks) < max_tracks:
                        new_track = StrictTrack(next_track_id, det, frame_idx)
                        active_tracks.append(new_track)
                        next_track_id += 1
                        stats['created'] += 1

        stats['max_concurrent'] = max(stats['max_concurrent'], len(active_tracks))

        # 可视化部分
        if show_live_video:
            visualize_strict_tracking(frame_idx, preprocessed_images_list, detections, active_tracks)

        # 进度
        if (frame_idx + 1) % 100 == 0:
            hq_count = sum(1 for t in active_tracks if t.is_high_quality())
            print(f"帧 {frame_idx + 1} | 活跃: {len(active_tracks)} (高质量: {hq_count}) | "
                  f"完成: {stats['completed']} | 拒绝匹配: {stats['rejected_matches']}")

    # 循环结束后关闭窗口
    if show_live_video:
        cv2.destroyAllWindows()

    # 添加剩余轨迹
    for track in active_tracks:
        if len(track.points) >= min_hits:
            completed_tracks.append(track)

    # 严格过滤
    final_tracks = []
    for track in completed_tracks:
        # # 过滤掉太长的轨迹（可能是跳跃形成的）
        # if (track.is_high_quality() and
        #         len(track.points) < 1500 and  # 不太可能有粒子被跟踪1500帧
        #         track.get_density() > 0.7):
        final_tracks.append(track)

    # 按长度排序，取最好的
    final_tracks.sort(key=lambda t: len(t.points), reverse=True)

    # 如果还是太多，只取最好的一部分
    if len(final_tracks) > 200:
        # 计算质量分数
        for track in final_tracks:
            track.quality_score = (
                    len(track.points) * 0.3 +
                    track.get_density() * 100 * 0.3 +
                    (50 - track.get_average_match_distance()) * 0.2 +
                    track.motion_consistency * 100 * 0.2
            )

        final_tracks.sort(key=lambda t: t.quality_score, reverse=True)
        final_tracks = final_tracks[:200]

    print(f"\n统计:")
    print(f"  创建: {stats['created']}")
    print(f"  完成: {stats['completed']}")
    print(f"  最终: {len(final_tracks)}")
    print(f"  拒绝的匹配: {stats['rejected_matches']}")

    return final_tracks


def visualize_strict_tracking(frame_idx, preprocessed_images_list, detections, active_tracks):
    """
    可视化函数，显示当前帧的检测点和活跃轨迹。
    """
    # 确保图像路径列表有效且当前帧索引在范围内
    if preprocessed_images_list and frame_idx < len(preprocessed_images_list):
        img_path = preprocessed_images_list[frame_idx]
        img = cv2.imread(img_path)
        if img is None:
            return

        # 绘制检测点
        for det in detections:
            cv2.circle(img, (int(det[0]), int(det[1])), 3, (0, 0, 255), -1)

        # 绘制轨迹
        for track in active_tracks:
            # 只绘制最近几帧内活跃的轨迹
            if track.last_frame_seen >= frame_idx - 10:
                points = track.get_ordered_points()
                if len(points) > 1:
                    # 根据质量选择颜色和线宽
                    if track.is_high_quality():
                        color = (0, 255, 0)  # 高质量轨迹为绿色
                        thickness = 2
                    else:
                        color = (0, 128, 255)  # 普通轨迹为橙色
                        thickness = 1

                    # 绘制最近的轨迹段，以保持画面清晰
                    recent_points = points[-30:] if len(points) > 30 else points
                    for i in range(len(recent_points) - 1):
                        p1 = tuple(map(int, recent_points[i]))
                        p2 = tuple(map(int, recent_points[i + 1]))
                        cv2.line(img, p1, p2, color, thickness)

                    # 标注轨迹ID
                    last_point = tuple(map(int, recent_points[-1]))
                    cv2.putText(img, f"{track.id}",
                                (last_point[0] + 5, last_point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 显示统计信息
        cv2.putText(img, f"Frame: {frame_idx} | Active Tracks: {len(active_tracks)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Strict Particle Tracking', img)
        cv2.waitKey(1)


def run_strict_tracking(det_left, det_right, out_left, out_right,
                        show_video=False, prep_left=None, prep_right=None):
    """运行严格跟踪"""

    # 严格参数
    params = {
        'max_age': 15,  # 很短的丢失容忍
        'min_hits_to_confirm': 20,  # 需要很多帧确认
        'dist_thresh': 40.0,  # 小的匹配距离
        'max_concurrent_tracks': 500  # 限制并发轨迹数
    }

    # 加载数据
    try:
        with open(det_left, 'rb') as f:
            detections_left = pickle.load(f)
        with open(det_right, 'rb') as f:
            detections_right = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 检测文件未找到。请确保文件路径正确。")
        return

    # 加载预处理图像文件列表
    prep_left_files = sorted(glob.glob(os.path.join(prep_left, '*.png'))) if prep_left else []
    prep_right_files = sorted(glob.glob(os.path.join(prep_right, '*.png'))) if prep_right else []

    print("\n=== 严格波浪粒子跟踪 ===")

    # 左侧
    print("\n--- 左侧相机跟踪 ---")
    tracks_left = strict_tracking(detections_left, params, prep_left_files, show_video)

    # 右侧
    print("\n--- 右侧相机跟踪 ---")
    tracks_right = strict_tracking(detections_right, params, prep_right_files, show_video)

    # 保存
    os.makedirs(os.path.dirname(out_left), exist_ok=True)
    with open(out_left, 'wb') as f:
        pickle.dump(tracks_left, f)
    os.makedirs(os.path.dirname(out_right), exist_ok=True)
    with open(out_right, 'wb') as f:
        pickle.dump(tracks_right, f)

    # 统计
    print("\n=== 最终结果 ===")
    for name, tracks in [("左侧", tracks_left), ("右侧", tracks_right)]:
        if tracks:
            lengths = [len(t.points) for t in tracks]
            densities = [t.get_density() for t in tracks]
            distances = [t.get_average_match_distance() for t in tracks]

            print(f"\n{name}: {len(tracks)} 条轨迹")
            print(f"  长度 - 平均: {np.mean(lengths):.1f}, "
                  f"中位数: {np.median(lengths):.1f}, "
                  f"最大: {max(lengths)}")
            print(f"  密度 - 平均: {np.mean(densities):.3f}")
            print(f"  匹配距离 - 平均: {np.mean(distances):.1f}")
            print(f"  >100帧: {sum(1 for l in lengths if l > 100)} 条")
            print(f"  >200帧: {sum(1 for l in lengths if l > 200)} 条")


if __name__ == '__main__':
    run_strict_tracking(
        "../data/detections/detections_left.pkl",
        "../data/detections/detections_right.pkl",
        "../data/trajectories/trajectories_2d_left_strict.pkl",
        "../data/trajectories/trajectories_2d_right_strict.pkl",
        show_video=True,
        prep_left="../data/preprocessed/left/",
        prep_right="../data/preprocessed/right/"
    )
