# particle_processing/03_trajectory_tracking_2d_ekf.py
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
import os
import cv2
import glob


class ImprovedKalmanFilter:
    """改进的卡尔曼滤波器，增强了对非线性运动的适应性"""

    def __init__(self, initial_pos):
        # 状态向量: [x, y, vx, vy, ax, ay] - 包含加速度
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0, 0.0], dtype=float)

        # 增大初始不确定性
        self.P = np.eye(6) * 1e-1
        self.P[4:6, 4:6] *= 10  # 加速度的初始不确定性更大

        # 过程噪声 - 适应椭圆运动
        self.Q = np.eye(6) * 1e-3
        self.Q[0:2, 0:2] *= 5  # 位置噪声
        self.Q[2:4, 2:4] *= 10  # 速度噪声
        self.Q[4:6, 4:6] *= 20  # 加速度噪声 - 允许更大的变化

        # 测量噪声
        self.R = np.eye(2) * 1e-1

        self.dt = 1.0

    def predict(self):
        # 使用恒加速度模型
        F = np.array([
            [1, 0, self.dt, 0, 0.5 * self.dt ** 2, 0],
            [0, 1, 0, self.dt, 0, 0.5 * self.dt ** 2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2]

    def update(self, m):
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        mv = np.array(m).reshape(2, 1)
        xv = self.x.reshape(6, 1)

        y = mv - H @ xv
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = (xv + K @ y).flatten()
        self.P = (np.eye(6) - K @ H) @ self.P


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器，用于处理椭圆运动"""

    def __init__(self, initial_pos):
        # 状态向量: [x, y, vx, vy, omega, phi]
        # omega: 角频率, phi: 相位
        self.x = np.array([
            initial_pos[0],
            initial_pos[1],
            0.0,  # vx
            0.0,  # vy
            0.1,  # omega (初始角频率估计)
            0.0  # phi (初始相位)
        ], dtype=float)

        # 协方差矩阵
        self.P = np.eye(6)
        self.P[0:2, 0:2] *= 1e-1  # 位置
        self.P[2:4, 2:4] *= 1e-1  # 速度
        self.P[4, 4] = 1e-2  # 角频率
        self.P[5, 5] = 1e-1  # 相位

        # 过程噪声
        self.Q = np.eye(6)
        self.Q[0:2, 0:2] *= 1e-3  # 位置噪声
        self.Q[2:4, 2:4] *= 1e-2  # 速度噪声
        self.Q[4, 4] = 1e-4  # 角频率变化很小
        self.Q[5, 5] = 1e-3  # 相位噪声

        # 测量噪声
        self.R = np.eye(2) * 1e-1

        self.dt = 1.0

        # 椭圆参数（可以根据实际情况调整）
        self.a = 50.0  # 椭圆长轴
        self.b = 30.0  # 椭圆短轴

    def f(self, x):
        """非线性状态转移函数"""
        x_new = x.copy()

        # 更新相位
        x_new[5] = x[5] + x[4] * self.dt

        # 椭圆运动模型
        # x = x0 + a * cos(phi)
        # y = y0 + b * sin(phi)
        # 这里假设椭圆中心缓慢漂移

        # 计算速度（椭圆运动的导数）
        x_new[2] = -self.a * x[4] * np.sin(x_new[5])
        x_new[3] = self.b * x[4] * np.cos(x_new[5])

        # 更新位置
        x_new[0] = x[0] + x_new[2] * self.dt
        x_new[1] = x[1] + x_new[3] * self.dt

        return x_new

    def compute_jacobian(self, x):
        """计算状态转移函数的雅可比矩阵"""
        F = np.eye(6)

        phi_new = x[5] + x[4] * self.dt

        # ∂x_new/∂x
        F[0, 2] = self.dt
        F[0, 4] = -self.a * self.dt * np.sin(phi_new) * self.dt
        F[0, 5] = -self.a * x[4] * self.dt * np.cos(phi_new)

        # ∂y_new/∂y
        F[1, 3] = self.dt
        F[1, 4] = self.b * self.dt * np.cos(phi_new) * self.dt
        F[1, 5] = -self.b * x[4] * self.dt * np.sin(phi_new)

        # ∂vx_new/∂...
        F[2, 4] = -self.a * np.sin(phi_new) - self.a * x[4] * self.dt * np.cos(phi_new)
        F[2, 5] = -self.a * x[4] * np.cos(phi_new)

        # ∂vy_new/∂...
        F[3, 4] = self.b * np.cos(phi_new) - self.b * x[4] * self.dt * np.sin(phi_new)
        F[3, 5] = -self.b * x[4] * np.sin(phi_new)

        # ∂phi_new/∂...
        F[5, 4] = self.dt

        return F

    def predict(self):
        """EKF预测步骤"""
        # 计算雅可比矩阵
        F = self.compute_jacobian(self.x)

        # 非线性状态预测
        self.x = self.f(self.x)

        # 协方差预测
        self.P = F @ self.P @ F.T + self.Q

        return self.x[:2]

    def update(self, m):
        """EKF更新步骤"""
        # 观测矩阵（线性）
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]])

        mv = np.array(m).reshape(2, 1)
        xv = self.x.reshape(6, 1)

        # 创新/残差
        y = mv - H @ xv

        # 创新协方差
        S = H @ self.P @ H.T + self.R

        # 卡尔曼增益
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状态更新
        self.x = (xv + K @ y).flatten()

        # 协方差更新
        self.P = (np.eye(6) - K @ H) @ self.P


class AdaptiveKalmanFilter:
    """自适应卡尔曼滤波器，可以动态调整噪声参数"""

    def __init__(self, initial_pos):
        self.kf = ImprovedKalmanFilter(initial_pos)
        self.innovation_history = []
        self.window_size = 10

    def predict(self):
        return self.kf.predict()

    def update(self, m):
        # 计算创新（预测误差）
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        predicted_measurement = H @ self.kf.x
        innovation = np.array(m) - predicted_measurement[:2]

        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)

        # 自适应调整测量噪声
        if len(self.innovation_history) >= 5:
            innovation_cov = np.cov(np.array(self.innovation_history).T)
            # 平滑更新R矩阵
            self.kf.R = 0.7 * self.kf.R + 0.3 * innovation_cov

        self.kf.update(m)


class Track:
    def __init__(self, track_id, initial_detection, frame_idx, filter_type='improved'):
        self.id = track_id

        # 选择滤波器类型

        if filter_type == 'improved':
            self.kf = ImprovedKalmanFilter(initial_detection)
        elif filter_type == 'ekf':
            self.kf = ExtendedKalmanFilter(initial_detection)
        elif filter_type == 'adaptive':
            self.kf = AdaptiveKalmanFilter(initial_detection)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        self.points = {frame_idx: initial_detection}
        self.age = 0
        self.total_visible_count = 1
        self.last_frame_seen = frame_idx
        self.velocity_history = []

    def predict(self):
        return self.kf.predict()

    def update(self, detection, frame_idx):
        # 计算速度用于分析
        if self.last_frame_seen in self.points:
            last_pos = np.array(self.points[self.last_frame_seen])
            current_pos = np.array(detection)
            dt = frame_idx - self.last_frame_seen
            if dt > 0:
                velocity = (current_pos - last_pos) / dt
                self.velocity_history.append(velocity)
                if len(self.velocity_history) > 20:
                    self.velocity_history.pop(0)

        self.kf.update(np.array(detection))
        self.points[frame_idx] = detection
        self.age = 0
        self.total_visible_count += 1
        self.last_frame_seen = frame_idx

    def get_ordered_points(self):
        return [self.points[fi] for fi in sorted(self.points.keys())]

    def mark_missed(self):
        self.age += 1

    def is_tentative(self, min_hits_for_confirmation=3):
        return self.total_visible_count < min_hits_for_confirmation

    def is_lost(self, max_age_limit=5):
        return self.age > max_age_limit


def track_particles_kalman_hungarian(all_detections_per_frame, max_age, min_hits_to_confirm, dist_thresh,
                                     preprocessed_images_list, show_live_video, filter_type='improved'):
    active_tracks, completed_tracks, next_track_id = [], [], 0
    total_frames = len(all_detections_per_frame)

    for frame_idx, detections_current_frame in enumerate(all_detections_per_frame):
        # 预测所有活跃轨迹的下一个位置
        predicted_positions = []
        for track in active_tracks:
            pred_pos = track.predict()
            predicted_positions.append(pred_pos)

        num_active_tracks, num_detections = len(active_tracks), len(detections_current_frame)
        cost_matrix = np.full((num_active_tracks, num_detections), np.inf)

        if num_active_tracks > 0 and num_detections > 0:
            for t_idx, pred_pos in enumerate(predicted_positions):
                for d_idx, det_pos in enumerate(detections_current_frame):
                    dist = np.linalg.norm(np.array(pred_pos) - np.array(det_pos))
                    if dist < dist_thresh:
                        cost_matrix[t_idx, d_idx] = dist

        matched_track_indices_set, matched_detection_indices_set = set(), set()
        try:
            if cost_matrix.size > 0 and np.any(np.isfinite(cost_matrix)):
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < dist_thresh:
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
                    if not track.is_tentative(min_hits_to_confirm):
                        completed_tracks.append(track)
                else:
                    new_active_tracks.append(track)
            else:
                new_active_tracks.append(track)
        active_tracks = new_active_tracks

        # 为未匹配的检测创建新轨迹
        for i, det in enumerate(detections_current_frame):
            if i not in matched_detection_indices_set:
                new_track = Track(next_track_id, det, frame_idx, filter_type)
                active_tracks.append(new_track)
                next_track_id += 1

        # 实时可视化
        if show_live_video and preprocessed_images_list:
            if frame_idx < len(preprocessed_images_list):
                img = cv2.imread(preprocessed_images_list[frame_idx])
                if img is not None:
                    # 绘制所有检测点
                    for det in detections_current_frame:
                        cv2.circle(img, (int(det[0]), int(det[1])), 3, (0, 0, 255), -1)

                    # 绘制所有活跃轨迹
                    for track in active_tracks:
                        points = track.get_ordered_points()
                        if len(points) > 1:
                            # 使用不同颜色表示不同的轨迹状态
                            color = (0, 255, 0) if not track.is_tentative() else (255, 255, 0)
                            for i in range(len(points) - 1):
                                p1 = tuple(map(int, points[i]))
                                p2 = tuple(map(int, points[i + 1]))
                                cv2.line(img, p1, p2, color, 2)

                            # 显示轨迹ID
                            last_point = tuple(map(int, points[-1]))
                            cv2.putText(img, str(track.id), last_point,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # 显示信息
                    info_text = f"Frame: {frame_idx} | Active: {len(active_tracks)} | Filter: {filter_type}"
                    cv2.putText(img, info_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    cv2.imshow('Live Tracking', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == total_frames:
            print(f"已处理 {frame_idx + 1}/{total_frames} 帧。当前活跃轨迹: {len(active_tracks)}")

    if show_live_video:
        cv2.destroyAllWindows()

    # 将剩余的活跃轨迹加入完成列表
    for track in active_tracks:
        if not track.is_tentative(min_hits_to_confirm):
            completed_tracks.append(track)

    return [t for t in completed_tracks if len(t.points) >= min_hits_to_confirm]


def run_tracking(detections_file_left, detections_file_right, output_traj_left, output_traj_right,
                 params_left, params_right, filter_type='improved', show_live_video=False,
                 preprocessed_dir_left=None, preprocessed_dir_right=None):
    """
    运行轨迹跟踪

    Args:
        filter_type: 'simple', 'improved', 'ekf', or 'adaptive'
    """
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

    print(f"\n--- 使用 {filter_type} 滤波器开始左侧相机的2D轨迹跟踪 ---")
    trajectories_left = track_particles_kalman_hungarian(
        all_detections_left,
        preprocessed_images_list=preprocessed_left_files,
        show_live_video=show_live_video,
        filter_type=filter_type,
        **params_left
    )
    print(f"为左侧相机找到 {len(trajectories_left)} 条轨迹。")

    print(f"\n--- 使用 {filter_type} 滤波器开始右侧相机的2D轨迹跟踪 ---")
    trajectories_right = track_particles_kalman_hungarian(
        all_detections_right,
        preprocessed_images_list=preprocessed_right_files,
        show_live_video=show_live_video,
        filter_type=filter_type,
        **params_right
    )
    print(f"为右侧相机找到 {len(trajectories_right)} 条轨迹。")

    # 保存轨迹
    os.makedirs(os.path.dirname(output_traj_left), exist_ok=True)
    with open(output_traj_left, 'wb') as f:
        pickle.dump(trajectories_left, f)
    print(f"\n左侧2D轨迹已保存至 {output_traj_left}")

    os.makedirs(os.path.dirname(output_traj_right), exist_ok=True)
    with open(output_traj_right, 'wb') as f:
        pickle.dump(trajectories_right, f)
    print(f"右侧2D轨迹已保存至 {output_traj_right}")

    # 打印一些统计信息
    print("\n--- 轨迹统计信息 ---")
    print(f"左侧相机: {len(trajectories_left)} 条轨迹")
    print(f"右侧相机: {len(trajectories_right)} 条轨迹")

    if trajectories_left:
        lengths_left = [len(t.points) for t in trajectories_left]
        print(f"左侧轨迹长度: 平均 {np.mean(lengths_left):.1f}, 最小 {min(lengths_left)}, 最大 {max(lengths_left)}")

    if trajectories_right:
        lengths_right = [len(t.points) for t in trajectories_right]
        print(f"右侧轨迹长度: 平均 {np.mean(lengths_right):.1f}, 最小 {min(lengths_right)}, 最大 {max(lengths_right)}")


if __name__ == '__main__':
    det_left_file = "../data/detections/detections_left.pkl"
    det_right_file = "../data/detections/detections_right.pkl"
    out_traj_l_file = "../data/trajectories/trajectories_2d_left.pkl"
    out_traj_r_file = "../data/trajectories/trajectories_2d_right.pkl"

    prep_left_dir = "../data/preprocessed/left/"
    prep_right_dir = "../data/preprocessed/right/"

    # 跟踪参数 - 针对椭圆运动进行了调整
    tracking_params = {
        'max_age': 30,  # 允许更长的消失时间
        'min_hits_to_confirm': 3,
        'dist_thresh': 150.0  # 增大阈值以适应椭圆运动
    }

    # 选择滤波器类型：'simple', 'improved', 'ekf', 'adaptive'
    filter_type = 'ekf'  # 建议先从improved开始，然后尝试ekf

    print(f"使用 {filter_type} 滤波器进行粒子跟踪...")

    run_tracking(
        det_left_file, det_right_file,
        out_traj_l_file, out_traj_r_file,
        tracking_params, tracking_params,
        filter_type=filter_type,
        show_live_video=True,
        preprocessed_dir_left=prep_left_dir,
        preprocessed_dir_right=prep_right_dir
    )