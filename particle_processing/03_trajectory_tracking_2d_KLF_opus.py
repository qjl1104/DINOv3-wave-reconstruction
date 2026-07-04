# particle_processing/03_trajectory_tracking_2d_optimized.py
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
import os
import cv2
import glob
from collections import deque


class OptimizedExtendedKalmanFilter:
    """优化的EKF，更适合实际波浪中的粒子运动"""

    def __init__(self, initial_pos):
        # 简化的状态向量: [x, y, vx, vy]
        # 去掉omega和phi，因为它们可能引入过多不确定性
        self.x = np.array([
            initial_pos[0],
            initial_pos[1],
            0.0,  # vx
            0.0  # vy
        ], dtype=float)

        # 协方差矩阵 - 增大初始不确定性
        self.P = np.eye(4)
        self.P[0:2, 0:2] *= 10.0  # 位置不确定性
        self.P[2:4, 2:4] *= 100.0  # 速度不确定性很大

        # 过程噪声 - 适应波浪运动的不规则性
        self.Q = np.eye(4)
        self.Q[0:2, 0:2] *= 5.0  # 位置过程噪声
        self.Q[2:4, 2:4] *= 50.0  # 速度过程噪声（波浪运动速度变化大）

        # 测量噪声
        self.R = np.eye(2) * 10.0  # 增大测量噪声，减少对单次测量的依赖

        self.dt = 1.0

        # 速度历史，用于自适应调整
        self.velocity_history = deque(maxlen=10)

    def predict(self):
        """使用简化的非线性模型"""
        # 简单的运动模型，但考虑速度衰减
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 0.95, 0],  # 速度衰减因子
            [0, 0, 0, 0.95]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        return self.x[:2]

    def update(self, m):
        """更新步骤，包含自适应调整"""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        mv = np.array(m).reshape(2, 1)
        xv = self.x.reshape(4, 1)

        # 计算新息
        y = mv - H @ xv

        # 自适应调整测量噪声
        innovation_magnitude = np.linalg.norm(y)
        if innovation_magnitude > 50:  # 如果新息很大，增加测量噪声
            R_adaptive = self.R * (1 + innovation_magnitude / 50)
        else:
            R_adaptive = self.R

        S = H @ self.P @ H.T + R_adaptive
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = (xv + K @ y).flatten()
        self.P = (np.eye(4) - K @ H) @ self.P

        # 记录速度历史
        self.velocity_history.append(self.x[2:4])


class RobustKalmanFilter:
    """鲁棒卡尔曼滤波器，使用多假设跟踪"""

    def __init__(self, initial_pos):
        # 状态向量: [x, y, vx, vy, ax, ay]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0, 0.0], dtype=float)

        # 使用多个运动模型
        self.models = {
            'constant_velocity': {'F': self._get_cv_matrix, 'Q': np.eye(6) * 10},
            'constant_acceleration': {'F': self._get_ca_matrix, 'Q': np.eye(6) * 5},
            'circular': {'F': self._get_circular_matrix, 'Q': np.eye(6) * 20}
        }

        # 模型概率
        self.model_probs = {
            'constant_velocity': 0.4,
            'constant_acceleration': 0.4,
            'circular': 0.2
        }

        self.P = np.eye(6) * 100
        self.R = np.eye(2) * 10
        self.dt = 1.0

    def _get_cv_matrix(self):
        """恒速模型"""
        F = np.eye(6)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        return F

    def _get_ca_matrix(self):
        """恒加速度模型"""
        F = np.eye(6)
        F[0, 2] = self.dt
        F[0, 4] = 0.5 * self.dt ** 2
        F[1, 3] = self.dt
        F[1, 5] = 0.5 * self.dt ** 2
        F[2, 4] = self.dt
        F[3, 5] = self.dt
        return F

    def _get_circular_matrix(self):
        """近似圆周运动模型"""
        F = self._get_ca_matrix()
        # 添加向心加速度的影响
        if np.linalg.norm(self.x[2:4]) > 0.1:
            v_norm = np.linalg.norm(self.x[2:4])
            F[4, 2] = -self.x[3] / v_norm * 0.1
            F[5, 3] = self.x[2] / v_norm * 0.1
        return F

    def predict(self):
        """交互式多模型预测"""
        predictions = {}
        covariances = {}

        for model_name, model in self.models.items():
            F = model['F']()
            Q = model['Q']

            x_pred = F @ self.x
            P_pred = F @ self.P @ F.T + Q

            predictions[model_name] = x_pred
            covariances[model_name] = P_pred

        # 混合预测
        self.x = np.zeros(6)
        self.P = np.zeros((6, 6))

        for model_name in self.models:
            self.x += self.model_probs[model_name] * predictions[model_name]

        for model_name in self.models:
            diff = predictions[model_name] - self.x
            self.P += self.model_probs[model_name] * (
                    covariances[model_name] + np.outer(diff, diff)
            )

        return self.x[:2]

    def update(self, m):
        """更新步骤，同时更新模型概率"""
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        mv = np.array(m).reshape(2, 1)
        xv = self.x.reshape(6, 1)

        y = mv - H @ xv
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = (xv + K @ y).flatten()
        self.P = (np.eye(6) - K @ H) @ self.P

        # 更新模型概率（简化版本）
        innovation_norm = np.linalg.norm(y)
        if innovation_norm < 10:
            self.model_probs['constant_velocity'] = 0.5
            self.model_probs['constant_acceleration'] = 0.4
            self.model_probs['circular'] = 0.1
        else:
            self.model_probs['constant_velocity'] = 0.2
            self.model_probs['constant_acceleration'] = 0.3
            self.model_probs['circular'] = 0.5


class Track:
    def __init__(self, track_id, initial_detection, frame_idx, filter_type='optimized_ekf'):
        self.id = track_id

        # 选择滤波器类型
        if filter_type == 'optimized_ekf':
            self.kf = OptimizedExtendedKalmanFilter(initial_detection)
        elif filter_type == 'robust':
            self.kf = RobustKalmanFilter(initial_detection)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        self.points = {frame_idx: initial_detection}
        self.predictions = {}  # 存储预测位置
        self.age = 0
        self.consecutive_misses = 0
        self.total_visible_count = 1
        self.last_frame_seen = frame_idx

        # 轨迹质量评分
        self.confidence = 1.0
        self.smoothness_score = 1.0

    def predict(self):
        pred_pos = self.kf.predict()
        return pred_pos

    def update(self, detection, frame_idx):
        # 计算预测误差
        if hasattr(self, 'last_prediction'):
            pred_error = np.linalg.norm(np.array(detection) - self.last_prediction)
            # 更新置信度
            if pred_error < 20:
                self.confidence = min(1.0, self.confidence * 1.1)
            else:
                self.confidence = max(0.1, self.confidence * 0.9)

        self.kf.update(np.array(detection))
        self.points[frame_idx] = detection
        self.age = 0
        self.consecutive_misses = 0
        self.total_visible_count += 1
        self.last_frame_seen = frame_idx

        # 计算平滑度
        self._update_smoothness_score()

    def _update_smoothness_score(self):
        """计算轨迹平滑度评分"""
        if len(self.points) >= 3:
            recent_points = list(self.points.values())[-5:]
            if len(recent_points) >= 3:
                # 计算加速度的变化
                accelerations = []
                for i in range(2, len(recent_points)):
                    p1 = np.array(recent_points[i - 2])
                    p2 = np.array(recent_points[i - 1])
                    p3 = np.array(recent_points[i])

                    v1 = p2 - p1
                    v2 = p3 - p2
                    acc = v2 - v1
                    accelerations.append(np.linalg.norm(acc))

                if accelerations:
                    avg_acc = np.mean(accelerations)
                    self.smoothness_score = 1.0 / (1.0 + avg_acc / 10.0)

    def mark_missed(self):
        self.age += 1
        self.consecutive_misses += 1
        self.confidence *= 0.95

    def get_ordered_points(self):
        return [self.points[fi] for fi in sorted(self.points.keys())]

    def is_tentative(self, min_hits_for_confirmation=3):
        return self.total_visible_count < min_hits_for_confirmation

    def is_lost(self, max_age_limit=5):
        # 考虑置信度的动态阈值
        dynamic_max_age = max_age_limit * (1 + self.confidence)
        return self.age > dynamic_max_age or self.consecutive_misses > max_age_limit * 2

    def get_quality_score(self):
        """综合质量评分"""
        length_score = min(1.0, len(self.points) / 100.0)
        visibility_score = self.total_visible_count / (self.last_frame_seen - min(self.points.keys()) + 1)
        return self.confidence * self.smoothness_score * length_score * visibility_score


def advanced_cost_matrix(tracks, detections, predicted_positions, dist_thresh):
    """高级成本矩阵计算，考虑多个因素"""
    num_tracks = len(tracks)
    num_detections = len(detections)
    cost_matrix = np.full((num_tracks, num_detections), np.inf)

    if num_tracks > 0 and num_detections > 0:
        for t_idx, (track, pred_pos) in enumerate(zip(tracks, predicted_positions)):
            for d_idx, det_pos in enumerate(detections):
                # 基础距离
                dist = np.linalg.norm(np.array(pred_pos) - np.array(det_pos))

                # 根据轨迹置信度调整阈值
                adaptive_thresh = dist_thresh * (1 + 0.5 * track.confidence)

                if dist < adaptive_thresh:
                    # 综合成本：距离 + 质量惩罚
                    quality_penalty = (1 - track.confidence) * 20
                    cost_matrix[t_idx, d_idx] = dist + quality_penalty

    return cost_matrix


def track_particles_optimized(all_detections_per_frame, max_age, min_hits_to_confirm, dist_thresh,
                              preprocessed_images_list, show_live_video, filter_type='optimized_ekf',
                              enable_track_merging=True):
    """优化的粒子跟踪函数"""
    active_tracks, completed_tracks, next_track_id = [], [], 0
    total_frames = len(all_detections_per_frame)

    # 跟踪统计
    stats = {
        'total_created': 0,
        'total_completed': 0,
        'total_merged': 0,
        'max_concurrent': 0
    }

    for frame_idx, detections_current_frame in enumerate(all_detections_per_frame):
        # 预测所有活跃轨迹的下一个位置
        predicted_positions = []
        for track in active_tracks:
            pred_pos = track.predict()
            track.last_prediction = pred_pos
            predicted_positions.append(pred_pos)

        # 计算高级成本矩阵
        cost_matrix = advanced_cost_matrix(active_tracks, detections_current_frame,
                                           predicted_positions, dist_thresh)

        matched_track_indices_set, matched_detection_indices_set = set(), set()

        # 匈牙利算法匹配
        try:
            if cost_matrix.size > 0 and np.any(np.isfinite(cost_matrix)):
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if np.isfinite(cost_matrix[r, c]):
                        active_tracks[r].update(detections_current_frame[c], frame_idx)
                        matched_track_indices_set.add(r)
                        matched_detection_indices_set.add(c)
        except ValueError:
            pass

        # 处理未匹配的轨迹
        new_active_tracks = []
        for i, track in enumerate(active_tracks):
            if i not in matched_track_indices_set:
                track.mark_missed()
                if track.is_lost(max_age):
                    if not track.is_tentative(min_hits_to_confirm) and track.get_quality_score() > 0.3:
                        completed_tracks.append(track)
                        stats['total_completed'] += 1
                else:
                    new_active_tracks.append(track)
            else:
                new_active_tracks.append(track)

        # 轨迹合并（可选）
        if enable_track_merging and len(new_active_tracks) > 1:
            merged_indices = set()
            for i in range(len(new_active_tracks)):
                if i in merged_indices:
                    continue
                for j in range(i + 1, len(new_active_tracks)):
                    if j in merged_indices:
                        continue

                    # 检查是否应该合并
                    track1, track2 = new_active_tracks[i], new_active_tracks[j]
                    if should_merge_tracks(track1, track2):
                        # 合并轨迹
                        merge_tracks(track1, track2)
                        merged_indices.add(j)
                        stats['total_merged'] += 1

            # 移除被合并的轨迹
            new_active_tracks = [t for i, t in enumerate(new_active_tracks) if i not in merged_indices]

        active_tracks = new_active_tracks

        # 为未匹配的检测创建新轨迹
        for i, det in enumerate(detections_current_frame):
            if i not in matched_detection_indices_set:
                # 检查是否太接近现有轨迹
                too_close = False
                for track in active_tracks:
                    if track.last_frame_seen == frame_idx:
                        last_pos = track.points[frame_idx]
                        if np.linalg.norm(np.array(det) - np.array(last_pos)) < 20:
                            too_close = True
                            break

                if not too_close:
                    new_track = Track(next_track_id, det, frame_idx, filter_type)
                    active_tracks.append(new_track)
                    next_track_id += 1
                    stats['total_created'] += 1

        # 更新统计
        stats['max_concurrent'] = max(stats['max_concurrent'], len(active_tracks))

        # 实时可视化
        if show_live_video and preprocessed_images_list:
            visualize_tracking(frame_idx, preprocessed_images_list, detections_current_frame,
                               active_tracks, stats, filter_type)

        # 进度报告
        if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == total_frames:
            print(f"已处理 {frame_idx + 1}/{total_frames} 帧。"
                  f"活跃: {len(active_tracks)}, "
                  f"完成: {stats['total_completed']}, "
                  f"合并: {stats['total_merged']}")

    if show_live_video:
        cv2.destroyAllWindows()

    # 将剩余的高质量活跃轨迹加入完成列表
    for track in active_tracks:
        if not track.is_tentative(min_hits_to_confirm) and track.get_quality_score() > 0.3:
            completed_tracks.append(track)

    # 过滤低质量轨迹
    high_quality_tracks = [t for t in completed_tracks
                           if len(t.points) >= min_hits_to_confirm * 2
                           and t.get_quality_score() > 0.5]

    print(f"\n跟踪统计:")
    print(f"总创建轨迹数: {stats['total_created']}")
    print(f"总完成轨迹数: {stats['total_completed']}")
    print(f"高质量轨迹数: {len(high_quality_tracks)}")
    print(f"总合并次数: {stats['total_merged']}")
    print(f"最大并发轨迹数: {stats['max_concurrent']}")

    return high_quality_tracks


def should_merge_tracks(track1, track2):
    """判断两条轨迹是否应该合并"""
    # 检查是否有重叠的时间范围
    frames1 = set(track1.points.keys())
    frames2 = set(track2.points.keys())

    if frames1.intersection(frames2):
        return False  # 有时间重叠，不能合并

    # 检查时间间隔
    if frames1 and frames2:
        gap = abs(max(frames1) - min(frames2))
        if gap > 10:  # 间隔太大
            return False

        # 检查空间距离
        if max(frames1) < min(frames2):
            last_point1 = track1.points[max(frames1)]
            first_point2 = track2.points[min(frames2)]
        else:
            last_point1 = track2.points[max(frames2)]
            first_point2 = track1.points[min(frames1)]

        distance = np.linalg.norm(np.array(last_point1) - np.array(first_point2))

        # 根据间隔时间调整距离阈值
        distance_threshold = 30 * (1 + gap / 5)

        return distance < distance_threshold

    return False


def merge_tracks(track1, track2):
    """合并两条轨迹"""
    # 合并点
    track1.points.update(track2.points)

    # 更新统计信息
    track1.total_visible_count += track2.total_visible_count
    track1.last_frame_seen = max(track1.last_frame_seen, track2.last_frame_seen)

    # 更新置信度
    track1.confidence = (track1.confidence + track2.confidence) / 2


def visualize_tracking(frame_idx, preprocessed_images_list, detections, active_tracks, stats, filter_type):
    """增强的可视化函数"""
    if frame_idx < len(preprocessed_images_list):
        img = cv2.imread(preprocessed_images_list[frame_idx])
        if img is not None:
            # 绘制所有检测点
            for det in detections:
                cv2.circle(img, (int(det[0]), int(det[1])), 3, (0, 0, 255), -1)

            # 绘制所有活跃轨迹
            for track in active_tracks:
                points = track.get_ordered_points()
                if len(points) > 1:
                    # 根据质量选择颜色
                    quality = track.get_quality_score()
                    if quality > 0.7:
                        color = (0, 255, 0)  # 绿色 - 高质量
                    elif quality > 0.4:
                        color = (0, 255, 255)  # 黄色 - 中等质量
                    else:
                        color = (0, 165, 255)  # 橙色 - 低质量

                    # 绘制轨迹线
                    for i in range(len(points) - 1):
                        p1 = tuple(map(int, points[i]))
                        p2 = tuple(map(int, points[i + 1]))
                        thickness = 2 if track.confidence > 0.5 else 1
                        cv2.line(img, p1, p2, color, thickness)

                    # 显示轨迹ID和置信度
                    last_point = tuple(map(int, points[-1]))
                    text = f"{track.id}:{track.confidence:.2f}"
                    cv2.putText(img, text, (last_point[0] + 5, last_point[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 显示统计信息
            y_offset = 30
            info_texts = [
                f"Frame: {frame_idx}",
                f"Active Tracks: {len(active_tracks)}",
                f"Detections: {len(detections)}",
                f"Filter: {filter_type}",
                f"Total Merged: {stats['total_merged']}"
            ]

            for i, text in enumerate(info_texts):
                cv2.putText(img, text, (10, y_offset + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Optimized Tracking', img)
            cv2.waitKey(1)


def run_tracking_optimized(detections_file_left, detections_file_right,
                           output_traj_left, output_traj_right,
                           params_left, params_right,
                           filter_type='optimized_ekf',
                           show_live_video=False,
                           preprocessed_dir_left=None,
                           preprocessed_dir_right=None):
    """运行优化的轨迹跟踪"""
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

    print(f"\n=== 使用优化的 {filter_type} 滤波器进行粒子跟踪 ===")

    print(f"\n--- 开始左侧相机的2D轨迹跟踪 ---")
    trajectories_left = track_particles_optimized(
        all_detections_left,
        preprocessed_images_list=preprocessed_left_files,
        show_live_video=show_live_video,
        filter_type=filter_type,
        enable_track_merging=True,
        **params_left
    )

    print(f"\n--- 开始右侧相机的2D轨迹跟踪 ---")
    trajectories_right = track_particles_optimized(
        all_detections_right,
        preprocessed_images_list=preprocessed_right_files,
        show_live_video=show_live_video,
        filter_type=filter_type,
        enable_track_merging=True,
        **params_right
    )

    # 保存轨迹
    os.makedirs(os.path.dirname(output_traj_left), exist_ok=True)
    with open(output_traj_left, 'wb') as f:
        pickle.dump(trajectories_left, f)
    print(f"\n左侧2D轨迹已保存至 {output_traj_left}")

    os.makedirs(os.path.dirname(output_traj_right), exist_ok=True)
    with open(output_traj_right, 'wb') as f:
        pickle.dump(trajectories_right, f)
    print(f"右侧2D轨迹已保存至 {output_traj_right}")

    # 详细统计信息
    print("\n=== 最终轨迹统计信息 ===")
    print(f"左侧相机高质量轨迹: {len(trajectories_left)} 条")
    print(f"右侧相机高质量轨迹: {len(trajectories_right)} 条")

    if trajectories_left:
        lengths_left = [len(t.points) for t in trajectories_left]
        qualities_left = [t.get_quality_score() for t in trajectories_left]
        print(f"\n左侧轨迹:")
        print(f"  长度 - 平均: {np.mean(lengths_left):.1f}, "
              f"最小: {min(lengths_left)}, 最大: {max(lengths_left)}")
        print(f"  质量 - 平均: {np.mean(qualities_left):.3f}, "
              f"最小: {min(qualities_left):.3f}, 最大: {max(qualities_left):.3f}")

    if trajectories_right:
        lengths_right = [len(t.points) for t in trajectories_right]
        qualities_right = [t.get_quality_score() for t in trajectories_right]
        print(f"\n右侧轨迹:")
        print(f"  长度 - 平均: {np.mean(lengths_right):.1f}, "
              f"最小: {min(lengths_right)}, 最大: {max(lengths_right)}")
        print(f"  质量 - 平均: {np.mean(qualities_right):.3f}, "
              f"最小: {min(qualities_right):.3f}, 最大: {max(qualities_right):.3f}")


if __name__ == '__main__':
    det_left_file = "../data/detections/detections_left.pkl"
    det_right_file = "../data/detections/detections_right.pkl"
    out_traj_l_file = "../data/trajectories/trajectories_2d_left_optimized.pkl"
    out_traj_r_file = "../data/trajectories/trajectories_2d_right_optimized.pkl"

    prep_left_dir = "../data/preprocessed/left/"
    prep_right_dir = "../data/preprocessed/right/"

    # 优化的跟踪参数
    tracking_params = {
        'max_age': 20,  # 减少到20，避免过长的预测
        'min_hits_to_confirm': 5,  # 增加到5，确保轨迹质量
        'dist_thresh': 80.0  # 减小到80，提高匹配精度
    }

    # 使用优化的EKF
    filter_type = 'optimized_ekf'  # 或 'robust'

    print(f"使用优化的 {filter_type} 滤波器进行粒子跟踪...")

    run_tracking_optimized(
        det_left_file, det_right_file,
        out_traj_l_file, out_traj_r_file,
        tracking_params, tracking_params,
        filter_type=filter_type,
        show_live_video=True,
        preprocessed_dir_left=prep_left_dir,
        preprocessed_dir_right=prep_right_dir
    )