# particle_processing/04_trajectory_matching.py
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
import os
import sys


# ----------------- 兼容性类定义 (从 03 脚本中复制) -----------------
# 这些类用于确保pickle.load能够正常工作，它们在匹配过程中不实际使用。
class SimpleKalmanFilter:
    def __init__(self, initial_pos): self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0],
                                                       dtype=float); self.P = np.eye(4) * 1e-1; self.Q = np.eye(
        4) * 1e-3; self.R = np.eye(2) * 1e-1; self.dt = 1.0

    def predict(self): F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0,
                                                                                            1]]); self.x = F @ self.x; self.P = F @ self.P @ F.T + self.Q; return self.x[
                                                                                                                                                                  :2]

    def update(self, m): H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]); mv = np.array(m).reshape(2,
                                                                                              1); xv = self.x.reshape(4,
                                                                                                                      1); y = mv - H @ xv; S = H @ self.P @ H.T + self.R; K = self.P @ H.T @ np.linalg.inv(
        S); self.x = (xv + K @ y).flatten(); self.P = (np.eye(4) - K @ H) @ self.P


class ImprovedKalmanFilter:
    def __init__(self, initial_pos): self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0, 0.0],
                                                       dtype=float); self.P = np.eye(6) * 1e-1; self.P[4:6,
                                                                                                4:6] *= 10; self.Q = np.eye(
        6) * 1e-3; self.Q[0:2, 0:2] *= 5; self.Q[2:4, 2:4] *= 10; self.Q[4:6, 4:6] *= 20; self.R = np.eye(
        2) * 1e-1; self.dt = 1.0

    def predict(self): F = np.array(
        [[1, 0, self.dt, 0, 0.5 * self.dt ** 2, 0], [0, 1, 0, self.dt, 0, 0.5 * self.dt ** 2], [0, 0, 1, 0, self.dt, 0],
         [0, 0, 0, 1, 0, self.dt], [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]); self.x = F @ self.x; self.P = F @ self.P @ F.T + self.Q; return self.x[:2]

    def update(self, m): H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]); mv = np.array(m).reshape(2,
                                                                                                          1); xv = self.x.reshape(
        6, 1); y = mv - H @ xv; S = H @ self.P @ H.T + self.R; K = self.P @ H.T @ np.linalg.inv(S); self.x = (
                xv + K @ y).flatten(); self.P = (np.eye(6) - K @ H) @ self.P


class ExtendedKalmanFilter:
    def __init__(self, initial_pos): self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.1, 0.0], dtype=float); self.P = np.eye(6); self.P[0:2, 0:2] *= 1e-1; self.P[2:4, 2:4] *= 1e-1; self.P[4, 4] = 1e-2; self.P[5, 5] = 1e-1; self.Q = np.eye(6); self.Q[0:2, 0:2] *= 1e-3; self.Q[2:4, 2:4] *= 1e-2; self.Q[4, 4] = 1e-4; self.Q[5, 5] = 1e-3; self.R = np.eye(2) * 1e-1; self.dt = 1.0; self.a = 50.0; self.b = 30.0
    def f(self, x):
        x_new = x.copy()
        x_new[5] = x[5] + x[4] * self.dt
        x_new[2] = -self.a * x[4] * np.sin(x_new[5])
        x_new[3] = self.b * x[4] * np.cos(x_new[5])
        x_new[0] = x[0] + x_new[2] * self.dt
        x_new[1] = x[1] + x_new[3] * self.dt
        return x_new
    def compute_jacobian(self, x): F = np.eye(6); phi_new = x[5] + x[4] * self.dt; F[0, 2] = self.dt; F[0, 4] = -self.a * self.dt * np.sin(phi_new) * self.dt; F[0, 5] = -self.a * x[4] * self.dt * np.cos(phi_new); F[1, 3] = self.dt; F[1, 4] = self.b * self.dt * np.cos(phi_new) * self.dt; F[1, 5] = -self.b * x[4] * self.dt * np.sin(phi_new); F[2, 4] = -self.a * np.sin(phi_new) - self.a * x[4] * self.dt * np.cos(phi_new); F[2, 5] = -self.a * x[4] * np.cos(phi_new); F[3, 4] = self.b * np.cos(phi_new) - self.b * x[4] * self.dt * np.sin(phi_new); F[3, 5] = -self.b * x[4] * np.sin(phi_new); F[5, 4] = self.dt; return F
    def predict(self): F = self.compute_jacobian(self.x); self.x = self.f(self.x); self.P = F @ self.P @ F.T + self.Q; return self.x[:2]
    def update(self, m): H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]); mv = np.array(m).reshape(2, 1); xv = self.x.reshape(6, 1); y = mv - H @ xv; S = H @ self.P @ H.T + self.R; K = self.P @ H.T @ np.linalg.inv(S); self.x = (xv + K @ y).flatten(); self.P = (np.eye(6) - K @ H) @ self.P

class AdaptiveKalmanFilter:
    def __init__(self, initial_pos):
        self.kf = ImprovedKalmanFilter(initial_pos)
        self.innovation_history = []
        self.window_size = 10

    def predict(self):
        return self.kf.predict()

    def update(self, m):
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        predicted_measurement = H @ self.kf.x
        innovation = np.array(m) - predicted_measurement[:2]

        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)

        if len(self.innovation_history) >= 5:
            innovation_cov = np.cov(np.array(self.innovation_history).T)
            self.kf.R = 0.7 * self.kf.R + 0.3 * innovation_cov

        self.kf.update(m)

# ----------------------------------------------------------------------------------


class Track:
    # 这里的__init__函数需要和03脚本中的保持一致
    def __init__(self, track_id, initial_detection, frame_idx):
        self.id = track_id
        # 为了兼容，这里默认使用SimpleKalmanFilter
        # 如果需要其他滤波器，需要在03脚本中修改
        self.kf = SimpleKalmanFilter(initial_detection)
        self.points = {frame_idx: initial_detection}
        self.age = 0
        self.total_visible_count = 1
        self.last_frame_seen = frame_idx

    def get_ordered_points(self):
        return [self.points[fi] for fi in sorted(self.points.keys())]


# ----------------- 新的 DFD 匹配函数和算法 -----------------

def discrete_frechet_distance(P, Q):
    """
    非递归实现的离散Fréchet距离算法 (动态规划).
    """
    n = len(P)
    m = len(Q)
    if n == 0 or m == 0:
        return np.inf

    ca = np.full((n, m), -1.0)

    # 初始化动态规划表
    ca[0, 0] = np.linalg.norm(P[0] - Q[0])

    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], np.linalg.norm(P[i] - Q[0]))

    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], np.linalg.norm(P[0] - Q[j]))

    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]),
                           np.linalg.norm(P[i] - Q[j]))

    return ca[n - 1, m - 1]


def verify_epipolar_constraint_points(points_left, points_right, F_matrix, threshold=1.5):
    # ... (与之前版本相同) ...
    num_check_points = min(len(points_left), len(points_right), 5)
    if num_check_points == 0: return False
    valid_count = 0
    for i in range(num_check_points):
        idx = i * (min(len(points_left), len(points_right)) // num_check_points)
        pt_l_tuple, pt_r_tuple = points_left[idx], points_right[idx]
        pt_l, pt_r = np.array([pt_l_tuple[0], pt_l_tuple[1], 1.0]), np.array([pt_r_tuple[0], pt_r_tuple[1], 1.0])
        line_r, line_l = F_matrix @ pt_l, F_matrix.T @ pt_r
        numerator = (pt_r.T @ F_matrix @ pt_l) ** 2
        denominator = line_r[0] ** 2 + line_r[1] ** 2 + line_l[0] ** 2 + line_l[1] ** 2
        if denominator > 1e-6 and (numerator / denominator) < threshold ** 2: valid_count += 1
    return (valid_count / num_check_points) > 0.6


def match_trajectories_dfd(trajectories_left, trajectories_right, F_matrix, params):
    if not trajectories_left or not trajectories_right: return []

    min_traj_len = params['min_traj_len']
    dfd_threshold = params['dfd_threshold']
    epipolar_thresh = params['epipolar_thresh']
    length_diff_ratio = params['length_diff_ratio']

    filtered_traj_left_obj = [t for t in trajectories_left if len(t.points) >= min_traj_len]
    filtered_traj_right_obj = [t for t in trajectories_right if len(t.points) >= min_traj_len]
    filtered_traj_left = [t.get_ordered_points() for t in filtered_traj_left_obj]
    filtered_traj_right = [t.get_ordered_points() for t in filtered_traj_right_obj]

    if not filtered_traj_left or not filtered_traj_right:
        print("没有足够长的轨迹进行匹配。");
        return []

    # --- 轨迹数据诊断 ---
    print("\n--- 轨迹数据诊断 ---")
    lengths_left = [len(t) for t in filtered_traj_left]
    lengths_right = [len(t) for t in filtered_traj_right]
    if lengths_left:
        print(
            f"左侧相机 (共 {len(lengths_left)} 条有效轨迹): 长度 最小值={np.min(lengths_left)}, 最大值={np.max(lengths_left)}, 平均值={np.mean(lengths_left):.1f}")
    else:
        print("左侧相机: 没有找到符合长度要求的轨迹。")
    if lengths_right:
        print(
            f"右侧相机 (共 {len(lengths_right)} 条有效轨迹): 长度 最小值={np.min(lengths_right)}, 最大值={np.max(lengths_right)}, 平均值={np.mean(lengths_right):.1f}")
    else:
        print("右侧相机: 没有找到符合长度要求的轨迹。")
    print("---------------------\n")
    # --- 诊断结束 ---

    features_left = [np.array(t, dtype=np.double) for t in filtered_traj_left]
    features_right = [np.array(t, dtype=np.double) for t in filtered_traj_right]
    num_left, num_right = len(features_left), len(features_right)
    dfd_cost_matrix = np.full((num_left, num_right), np.inf)

    skipped_by_length_count = 0
    dfd_computed_count = 0

    print(f"正在为 {num_left} 条左轨迹和 {num_right} 条右轨迹构建DFD成本矩阵...")
    # 增加的预过滤，用于避免昂贵的DFD计算
    pre_filter_dist_thresh = 200.0  # 预过滤距离阈值，可调

    for i in range(num_left):
        for j in range(num_right):
            len_l, len_r = len(features_left[i]), len(features_right[j])

            # 1. 轨迹长度差异过滤
            if abs(len_l - len_r) > max(len_l, len_r) * length_diff_ratio:
                skipped_by_length_count += 1
                continue

            # 2. 新增：基于起始点和结束点的欧氏距离进行过滤
            if pre_filter_dist_thresh > 0:
                start_dist = np.linalg.norm(features_left[i][0] - features_right[j][0])
                end_dist = np.linalg.norm(features_left[i][-1] - features_right[j][-1])
                if start_dist > pre_filter_dist_thresh or end_dist > pre_filter_dist_thresh:
                    skipped_by_length_count += 1
                    continue

            dfd_computed_count += 1
            try:
                distance = discrete_frechet_distance(features_left[i], features_right[j])
                dfd_cost_matrix[i, j] = distance
                if dfd_computed_count <= 5:
                    print(
                        f"  > (调试) 计算 DFD: 左轨迹 {i} (长 {len_l}) vs 右轨迹 {j} (长 {len_r}). 距离 = {distance:.2f}")
            except (RecursionError, Exception) as e:
                # 递归深度限制可能导致RecursionError
                print(f"警告: DFD计算失败 ({e})，轨迹 {i} vs {j}")
                pass

        if (i + 1) % 100 == 0 or (i + 1) == num_left:
            print(f"  已完成左侧轨迹 {i + 1}/{num_left} 的DFD计算。")

    print("\n--- 匹配过程诊断报告 ---")
    print(f"总计可能的配对数: {num_left * num_right}")
    print(f"因预过滤被跳过的配对数: {skipped_by_length_count}")
    print(f"尝试计算DFD的配对数: {dfd_computed_count}")
    print("--------------------------\n")

    if not np.any(np.isfinite(dfd_cost_matrix)):
        print("DFD成本矩阵中没有有效值，无法进行匹配。请检查诊断报告，确认问题所在。");
        return []

    row_ind, col_ind = linear_sum_assignment(dfd_cost_matrix)
    matched_pairs_indices = []
    for r, c in zip(row_ind, col_ind):
        if dfd_cost_matrix[r, c] < dfd_threshold:
            if verify_epipolar_constraint_points(filtered_traj_left[r], filtered_traj_right[c], F_matrix,
                                                 epipolar_thresh):
                matched_pairs_indices.append((r, c))

    print(f"经过DFD和对极约束验证后，找到 {len(matched_pairs_indices)} 对匹配的轨迹。")
    final_matched_pairs = [(filtered_traj_left_obj[r], filtered_traj_right_obj[c]) for r, c in matched_pairs_indices]
    return final_matched_pairs


def run_trajectory_matching(traj_file_left, traj_file_right, calib_params_file, output_matched_file, matching_params):
    try:
        # 为了兼容，需要将所有可能使用的类定义导入
        with open(traj_file_left, 'rb') as f:
            trajectories_l = pickle.load(f)
        with open(traj_file_right, 'rb') as f:
            trajectories_r = pickle.load(f)
        calib_data = np.load(calib_params_file)
        if 'F' not in calib_data: print(f"错误: 'F'矩阵未在 {calib_params_file} 中找到。"); return
        F_matrix = calib_data['F']
    except FileNotFoundError:
        print(f"错误: 文件未找到。");
        return
    except (AttributeError, ImportError) as e:
        print(
            f"错误: 加载轨迹文件失败，可能缺少必要的类定义。请确保04_trajectory_matching.py包含了03脚本中的所有滤波器类定义。")
        print(f"详细错误信息: {e}")
        return

    matched_trajectory_pairs = match_trajectories_dfd(trajectories_l, trajectories_r, F_matrix, matching_params)
    if matched_trajectory_pairs is None: print("未找到任何匹配的轨迹对。"); return

    os.makedirs(os.path.dirname(output_matched_file), exist_ok=True)
    with open(output_matched_file, 'wb') as f:
        pickle.dump(matched_trajectory_pairs, f)
    print(f"匹配的2D轨迹对已保存至 {output_matched_file}")


if __name__ == '__main__':
    traj_l_file = "../data/trajectories/trajectories_2d_left.pkl"
    traj_r_file = "../data/trajectories/trajectories_2d_right.pkl"
    calib_file = "../camera_calibration/params/stereo_calib_params_from_matlab_full.npz"
    out_matched_pkl = "../data/trajectories/matched_pairs_2d.pkl"

    matching_parameters = {
        'min_traj_len': 4,
        'dfd_threshold': 200.0,  # DFD阈值，需要根据实际情况调整
        'epipolar_thresh': 2.5,
        'length_diff_ratio': 0.50  # 减小该值以更严格地过滤
    }

    run_trajectory_matching(traj_l_file, traj_r_file, calib_file, out_matched_pkl, matching_parameters)