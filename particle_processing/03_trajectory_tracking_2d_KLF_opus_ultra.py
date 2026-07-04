# particle_processing/03_trajectory_tracking_2d_wave_optimized.py
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
import os
import cv2
import glob
from collections import deque, defaultdict


class WaveParticleKalmanFilter:
    """专门为波浪中粒子设计的卡尔曼滤波器"""

    def __init__(self, initial_pos):
        # 状态向量: [x, y, vx, vy]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)

        # 协方差矩阵 - 适合椭圆运动
        self.P = np.eye(4)
        self.P[0:2, 0:2] *= 10.0  # 位置不确定性（增加）
        self.P[2:4, 2:4] *= 30.0  # 速度不确定性（增加）

        # 过程噪声 - 针对周期性运动调整
        self.Q = np.eye(4)
        self.Q[0:2, 0:2] *= 5.0  # 位置噪声
        self.Q[2:4, 2:4] *= 25.0  # 速度噪声（增加）

        # 测量噪声
        self.R = np.eye(2) * 10.0

        self.dt = 1.0

        # 运动历史，用于学习运动模式
        self.position_history = deque(maxlen=20)
        self.velocity_history = deque(maxlen=10)
        self.position_history.append(initial_pos)

    def predict(self):
        """预测步骤"""
        # 基础线性模型
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 0.95, 0],  # 轻微速度衰减
            [0, 0, 0, 0.95]
        ])

        # 如果有足够的历史数据，尝试检测周期性运动
        if len(self.position_history) >= 10:
            # 计算最近的平均速度方向变化
            recent_positions = list(self.position_history)[-10:]
            if self._detect_circular_motion(recent_positions):
                # 如果检测到圆周运动，调整预测
                F[2:4, 2:4] *= 0.9  # 更强的速度衰减

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        return self.x[:2]

    def update(self, m):
        """更新步骤"""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        mv = np.array(m).reshape(2, 1)
        xv = self.x.reshape(4, 1)

        # 计算新息
        y = mv - H @ xv
        innovation_norm = np.linalg.norm(y)

        # 自适应测量噪声
        if innovation_norm > 50:  # 大的新息可能意味着错误匹配
            R_adaptive = self.R * 2.0
        else:
            R_adaptive = self.R

        S = H @ self.P @ H.T + R_adaptive
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = (xv + K @ y).flatten()
        self.P = (np.eye(4) - K @ H) @ self.P

        # 更新历史
        self.position_history.append(m)
        if len(self.position_history) >= 2:
            velocity = np.array(self.position_history[-1]) - np.array(self.position_history[-2])
            self.velocity_history.append(velocity)

    def _detect_circular_motion(self, positions):
        """检测是否为圆周/椭圆运动"""
        if len(positions) < 5:
            return False

        # 计算连续的速度向量
        velocities = []
        for i in range(1, len(positions)):
            v = np.array(positions[i]) - np.array(positions[i - 1])
            if np.linalg.norm(v) > 0:
                velocities.append(v / np.linalg.norm(v))

        if len(velocities) < 3:
            return False

        # 计算速度方向的变化
        angle_changes = []
        for i in range(1, len(velocities)):
            cos_angle = np.dot(velocities[i - 1], velocities[i])
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angle_changes.append(angle)

        # 如果角度变化相对恒定，可能是圆周运动
        if angle_changes:
            mean_change = np.mean(angle_changes)
            std_change = np.std(angle_changes)
            # 角度变化在一定范围内且相对稳定
            return 0.1 < mean_change < 0.5 and std_change < 0.3

        return False


class WaveParticleTrack:
    """波浪粒子轨迹类"""

    def __init__(self, track_id, initial_detection, frame_idx):
        self.id = track_id
        self.kf = WaveParticleKalmanFilter(initial_detection)
        self.points = {frame_idx: initial_detection}
        self.predictions = {}

        # 状态管理
        self.age = 0
        self.consecutive_misses = 0
        self.total_visible_count = 1
        self.last_frame_seen = frame_idx
        self.birth_frame = frame_idx

        # 质量指标
        self.confidence = 1.0
        self.match_history = deque(maxlen=10)  # 记录匹配距离

        # 运动特征
        self.is_elliptical = False
        self.ellipse_params = None

    def predict(self):
        pred_pos = self.kf.predict()
        self.predictions[self.last_frame_seen + 1] = pred_pos
        return pred_pos

    def update(self, detection, frame_idx, match_distance):
        # 记录匹配距离
        self.match_history.append(match_distance)

        # 更新置信度
        if match_distance < 30:
            self.confidence = min(1.0, self.confidence * 1.02)
        elif match_distance > 60:
            self.confidence = max(0.1, self.confidence * 0.95)
        else:
            self.confidence = max(0.5, self.confidence * 0.99)

        # 更新卡尔曼滤波器
        self.kf.update(np.array(detection))
        self.points[frame_idx] = detection

        # 更新状态
        self.age = 0
        self.consecutive_misses = 0
        self.total_visible_count += 1
        self.last_frame_seen = frame_idx

        # 检测椭圆运动
        if len(self.points) >= 10:
            self._detect_elliptical_motion()

    def _detect_elliptical_motion(self):
        """检测是否为椭圆运动"""
        recent_points = list(self.points.values())[-15:]
        if len(recent_points) < 10:
            return

        # 计算中心点
        center = np.mean(recent_points, axis=0)

        # 计算到中心的距离
        distances = [np.linalg.norm(p - center) for p in recent_points]

        # 如果距离变化有周期性，可能是椭圆运动
        if np.std(distances) < np.mean(distances) * 0.5:
            self.is_elliptical = True

    def mark_missed(self):
        self.age += 1
        self.consecutive_misses += 1
        self.confidence *= 0.95

    def get_ordered_points(self):
        return [self.points[fi] for fi in sorted(self.points.keys())]

    def get_lifetime(self, current_frame):
        return current_frame - self.birth_frame

    def get_density(self):
        if self.last_frame_seen >= self.birth_frame:
            lifetime = self.last_frame_seen - self.birth_frame + 1
            return self.total_visible_count / lifetime if lifetime > 0 else 1.0
        return 1.0

    def is_high_quality(self):
        """判断是否为高质量轨迹"""
        avg_match_dist = np.mean(self.match_history) if self.match_history else float('inf')
        return (self.confidence > 0.4 and
                self.get_density() > 0.5 and
                len(self.points) >= 20 and
                avg_match_dist < 40)

    def should_terminate(self, max_age):
        """决定是否终止轨迹"""
        # 基础条件
        if self.consecutive_misses > max_age:
            return True

        # 低置信度轨迹更早终止
        if self.confidence < 0.3 and self.consecutive_misses > max_age // 2:
            return True

        # 如果检测到椭圆运动，给更多机会
        if self.is_elliptical and self.consecutive_misses < max_age * 2:
            return False

        return False


def wave_optimized_tracking(all_detections_per_frame, max_age, min_hits_to_confirm,
                            dist_thresh, preprocessed_images_list, show_live_video):
    """针对波浪粒子优化的跟踪算法"""

    active_tracks = []
    completed_tracks = []
    next_track_id = 0
    total_frames = len(all_detections_per_frame)

    # 统计
    stats = {
        'total_created': 0,
        'total_completed': 0,
        'total_terminated': 0,
        'max_concurrent': 0,
        'wrong_matches': 0
    }

    for frame_idx, detections_current_frame in enumerate(all_detections_per_frame):
        # 预测所有活跃轨迹
        predicted_positions = []
        for track in active_tracks:
            pred_pos = track.predict()
            predicted_positions.append(pred_pos)

        # 匹配
        matched_track_indices = set()
        matched_detection_indices = set()

        if len(active_tracks) > 0 and len(detections_current_frame) > 0:
            # 计算成本矩阵
            cost_matrix = np.full((len(active_tracks), len(detections_current_frame)), 1000.0)  # 使用大值而不是inf

            for t_idx, (track, pred_pos) in enumerate(zip(active_tracks, predicted_positions)):
                for d_idx, det in enumerate(detections_current_frame):
                    dist = np.linalg.norm(pred_pos - np.array(det))

                    # 根据轨迹特性调整阈值
                    adaptive_thresh = dist_thresh

                    # 如果是椭圆运动，允许稍大的距离
                    if track.is_elliptical:
                        adaptive_thresh *= 1.2

                    # 根据置信度调整
                    if track.confidence > 0.7:
                        adaptive_thresh *= 1.1
                    elif track.confidence < 0.3:
                        adaptive_thresh *= 0.7

                    # 考虑历史匹配距离
                    if track.match_history:
                        avg_hist_dist = np.mean(track.match_history)
                        if dist > avg_hist_dist * 3:  # 距离突然变大，可能是错误匹配
                            adaptive_thresh *= 0.5

                    if dist < adaptive_thresh:
                        cost_matrix[t_idx, d_idx] = dist

            # 匈牙利算法 - 添加错误处理
            if np.any(np.isfinite(cost_matrix)):
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    for r, c in zip(row_ind, col_ind):
                        if np.isfinite(cost_matrix[r, c]) and cost_matrix[r, c] < dist_thresh * 2:
                            match_dist = cost_matrix[r, c]

                            # 额外的匹配验证
                            track = active_tracks[r]
                            if track.match_history and len(track.match_history) >= 3:
                                avg_dist = np.mean(list(track.match_history)[-3:])
                                # 如果匹配距离突然增大很多，可能是错误
                                if match_dist > avg_dist * 3.0 and match_dist > 50:
                                    stats['wrong_matches'] += 1
                                    continue

                            track.update(detections_current_frame[c], frame_idx, match_dist)
                            matched_track_indices.add(r)
                            matched_detection_indices.add(c)
                except ValueError as e:
                    # 如果匈牙利算法失败，使用贪心匹配
                    if frame_idx % 100 == 0:  # 只在某些帧打印，避免过多输出
                        print(f"Warning: Hungarian algorithm failed at frame {frame_idx}, using greedy matching")
                    # 贪心匹配：为每个轨迹找最近的检测
                    for t_idx in range(len(active_tracks)):
                        best_d_idx = -1
                        best_dist = float('inf')

                        for d_idx in range(len(detections_current_frame)):
                            if d_idx not in matched_detection_indices and cost_matrix[t_idx, d_idx] < best_dist:
                                best_dist = cost_matrix[t_idx, d_idx]
                                best_d_idx = d_idx

                        if best_d_idx >= 0 and best_dist < dist_thresh:
                            track = active_tracks[t_idx]
                            track.update(detections_current_frame[best_d_idx], frame_idx, best_dist)
                            matched_track_indices.add(t_idx)
                            matched_detection_indices.add(best_d_idx)

        # 处理未匹配的轨迹
        new_active_tracks = []
        for i, track in enumerate(active_tracks):
            if i not in matched_track_indices:
                track.mark_missed()

                if track.should_terminate(max_age):
                    if len(track.points) >= min_hits_to_confirm:
                        completed_tracks.append(track)
                        stats['total_completed'] += 1
                    else:
                        stats['total_terminated'] += 1
                else:
                    new_active_tracks.append(track)
            else:
                new_active_tracks.append(track)

        active_tracks = new_active_tracks

        # 创建新轨迹 - 更严格的条件
        for i, det in enumerate(detections_current_frame):
            if i not in matched_detection_indices:
                # 检查是否离现有轨迹太近
                too_close = False
                for track in active_tracks:
                    if track.last_frame_seen >= frame_idx - 1:  # 最近1帧内活跃
                        last_pos = track.points.get(track.last_frame_seen)
                        if last_pos is not None:
                            if np.linalg.norm(np.array(det) - np.array(last_pos)) < 40:
                                too_close = True
                                break

                if not too_close:
                    new_track = WaveParticleTrack(next_track_id, det, frame_idx)
                    active_tracks.append(new_track)
                    next_track_id += 1
                    stats['total_created'] += 1

        # 更新统计
        stats['max_concurrent'] = max(stats['max_concurrent'], len(active_tracks))

        # 可视化
        if show_live_video and preprocessed_images_list:
            visualize_wave_tracking(frame_idx, preprocessed_images_list, detections_current_frame,
                                    active_tracks, matched_detection_indices, stats)

        # 进度报告
        if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == total_frames:
            high_quality_count = sum(1 for t in active_tracks if t.is_high_quality())
            print(f"帧 {frame_idx + 1}/{total_frames} | "
                  f"活跃: {len(active_tracks)} (高质量: {high_quality_count}) | "
                  f"完成: {stats['total_completed']} | "
                  f"错误匹配: {stats['wrong_matches']}")

    if show_live_video:
        cv2.destroyAllWindows()

    # 添加剩余的高质量轨迹
    for track in active_tracks:
        if len(track.points) >= min_hits_to_confirm:
            completed_tracks.append(track)

    # 最终过滤 - 只保留高质量的长轨迹
    final_tracks = []
    for track in completed_tracks:
        if track.is_high_quality() or (len(track.points) >= 50 and track.get_density() > 0.7):
            final_tracks.append(track)

    print(f"\n=== 波浪粒子跟踪统计 ===")
    print(f"总创建轨迹: {stats['total_created']}")
    print(f"总完成轨迹: {stats['total_completed']}")
    print(f"最终高质量轨迹: {len(final_tracks)}")
    print(f"检测到的错误匹配: {stats['wrong_matches']}")

    return final_tracks


def visualize_wave_tracking(frame_idx, preprocessed_images_list, detections,
                            active_tracks, matched_indices, stats):
    """可视化函数"""
    if frame_idx < len(preprocessed_images_list):
        img = cv2.imread(preprocessed_images_list[frame_idx])
        if img is not None:
            # 绘制未匹配的检测（红色）
            for i, det in enumerate(detections):
                if i not in matched_indices:
                    cv2.circle(img, (int(det[0]), int(det[1])), 3, (0, 0, 255), -1)
                else:
                    cv2.circle(img, (int(det[0]), int(det[1])), 3, (0, 255, 0), -1)

            # 绘制轨迹
            for track in active_tracks:
                points = track.get_ordered_points()
                if len(points) > 1:
                    # 根据质量选择颜色
                    if track.is_high_quality():
                        color = (0, 255, 0)  # 绿色
                        thickness = 2
                    elif track.is_elliptical:
                        color = (255, 255, 0)  # 青色 - 椭圆运动
                        thickness = 2
                    else:
                        color = (128, 128, 255)  # 浅紫色
                        thickness = 1

                    # 只绘制最近的轨迹段，避免混乱
                    recent_points = points[-30:] if len(points) > 30 else points
                    for i in range(len(recent_points) - 1):
                        p1 = tuple(map(int, recent_points[i]))
                        p2 = tuple(map(int, recent_points[i + 1]))
                        cv2.line(img, p1, p2, color, thickness)

                    # 标注长轨迹
                    if len(points) > 50:
                        last_point = tuple(map(int, points[-1]))
                        cv2.putText(img, f"{track.id}:{len(points)}",
                                    (last_point[0] + 5, last_point[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 显示统计
            cv2.putText(img, f"Frame: {frame_idx} | Active: {len(active_tracks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Wave Particle Tracking', img)
            cv2.waitKey(1)


def run_wave_optimized_tracking(detections_file_left, detections_file_right,
                                output_traj_left, output_traj_right,
                                show_live_video=False,
                                preprocessed_dir_left=None,
                                preprocessed_dir_right=None):
    """运行波浪优化的跟踪"""

    # 波浪粒子专用参数
    tracking_params = {
        'max_age': 15,  # 稍微增加丢失容忍度
        'min_hits_to_confirm': 10,  # 需要更多帧来确认轨迹
        'dist_thresh': 80.0  # 增加匹配距离阈值到80
    }

    try:
        with open(detections_file_left, 'rb') as f:
            all_detections_left = pickle.load(f)
        with open(detections_file_right, 'rb') as f:
            all_detections_right = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 检测文件未找到。")
        return

    preprocessed_left_files = sorted(
        glob.glob(os.path.join(preprocessed_dir_left, '*.png'))) if preprocessed_dir_left else []
    preprocessed_right_files = sorted(
        glob.glob(os.path.join(preprocessed_dir_right, '*.png'))) if preprocessed_dir_right else []

    print("\n=== 波浪粒子专用跟踪系统 ===")

    print("\n--- 左侧相机跟踪 ---")
    trajectories_left = wave_optimized_tracking(
        all_detections_left,
        preprocessed_images_list=preprocessed_left_files,
        show_live_video=show_live_video,
        **tracking_params
    )

    print("\n--- 右侧相机跟踪 ---")
    trajectories_right = wave_optimized_tracking(
        all_detections_right,
        preprocessed_images_list=preprocessed_right_files,
        show_live_video=show_live_video,
        **tracking_params
    )

    # 保存结果
    os.makedirs(os.path.dirname(output_traj_left), exist_ok=True)
    with open(output_traj_left, 'wb') as f:
        pickle.dump(trajectories_left, f)

    os.makedirs(os.path.dirname(output_traj_right), exist_ok=True)
    with open(output_traj_right, 'wb') as f:
        pickle.dump(trajectories_right, f)

    # 详细统计
    print("\n=== 最终统计 ===")
    print(f"左侧高质量轨迹: {len(trajectories_left)} 条")
    print(f"右侧高质量轨迹: {len(trajectories_right)} 条")

    if trajectories_left:
        lengths = [len(t.points) for t in trajectories_left]
        print(f"\n左侧轨迹长度:")
        print(f"  平均: {np.mean(lengths):.1f}")
        print(f"  中位数: {np.median(lengths):.1f}")
        print(f"  最大: {max(lengths)}")
        print(f"  >100帧: {sum(1 for l in lengths if l > 100)} 条")
        print(f"  >200帧: {sum(1 for l in lengths if l > 200)} 条")

    if trajectories_right:
        lengths = [len(t.points) for t in trajectories_right]
        print(f"\n右侧轨迹长度:")
        print(f"  平均: {np.mean(lengths):.1f}")
        print(f"  中位数: {np.median(lengths):.1f}")
        print(f"  最大: {max(lengths)}")
        print(f"  >100帧: {sum(1 for l in lengths if l > 100)} 条")
        print(f"  >200帧: {sum(1 for l in lengths if l > 200)} 条")


if __name__ == '__main__':
    det_left_file = "../data/detections/detections_left.pkl"
    det_right_file = "../data/detections/detections_right.pkl"
    out_traj_l_file = "../data/trajectories/trajectories_2d_left_wave.pkl"
    out_traj_r_file = "../data/trajectories/trajectories_2d_right_wave.pkl"

    prep_left_dir = "../data/preprocessed/left/"
    prep_right_dir = "../data/preprocessed/right/"

    print("运行波浪粒子专用跟踪系统...")

    run_wave_optimized_tracking(
        det_left_file, det_right_file,
        out_traj_l_file, out_traj_r_file,
        show_live_video=True,
        preprocessed_dir_left=prep_left_dir,
        preprocessed_dir_right=prep_right_dir
    )