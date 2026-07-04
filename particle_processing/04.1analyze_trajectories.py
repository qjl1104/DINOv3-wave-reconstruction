# particle_processing/04_trajectory_matching.py
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
from dtaidistance import dtw
import os


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


class Track:
    def __init__(self, track_id, initial_detection, frame_idx): self.id = track_id; self.kf = SimpleKalmanFilter(
        initial_detection); self.points = {
        frame_idx: initial_detection}; self.age = 0; self.total_visible_count = 1; self.last_frame_seen = frame_idx

    def get_ordered_points(self): return [self.points[fi] for fi in sorted(self.points.keys())]


def verify_epipolar_constraint_points(points_left, points_right, F_matrix, threshold=1.5):
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


def match_trajectories_dtw(trajectories_left, trajectories_right, F_matrix, params):
    if not trajectories_left or not trajectories_right: return []

    min_traj_len = params['min_traj_len']
    dtw_threshold = params['dtw_threshold']
    epipolar_thresh = params['epipolar_thresh']
    length_diff_ratio = params['length_diff_ratio']

    filtered_traj_left_obj = [t for t in trajectories_left if len(t.points) >= min_traj_len]
    filtered_traj_right_obj = [t for t in trajectories_right if len(t.points) >= min_traj_len]
    filtered_traj_left = [t.get_ordered_points() for t in filtered_traj_left_obj]
    filtered_traj_right = [t.get_ordered_points() for t in filtered_traj_right_obj]

    if not filtered_traj_left or not filtered_traj_right:
        print("没有足够长的轨迹进行匹配。");
        return []

    features_left = [np.array(t, dtype=np.double) for t in filtered_traj_left]
    features_right = [np.array(t, dtype=np.double) for t in filtered_traj_right]
    num_left, num_right = len(features_left), len(features_right)
    dtw_cost_matrix = np.full((num_left, num_right), np.inf)

    print(f"正在为 {num_left} 条左轨迹和 {num_right} 条右轨迹构建DTW成本矩阵...")
    for i in range(num_left):
        for j in range(num_right):
            len_l, len_r = len(features_left[i]), len(features_right[j])
            if abs(len_l - len_r) > max(len_l, len_r) * length_diff_ratio: continue
            window_size = int(max(len_l, len_r) * 0.2);
            window_size = max(1, window_size)
            try:
                distance = dtw.distance_fast(features_left[i], features_right[j], use_pruning=True, window=window_size)
                dtw_cost_matrix[i, j] = distance
            except Exception:
                pass
        if (i + 1) % 100 == 0 or (i + 1) == num_left:
            print(f"  已完成左侧轨迹 {i + 1}/{num_left} 的DTW计算。")

    if not np.any(np.isfinite(dtw_cost_matrix)):
        print("DTW成本矩阵中没有有效值，无法进行匹配。");
        return []

    row_ind, col_ind = linear_sum_assignment(dtw_cost_matrix)
    matched_pairs_indices = []
    for r, c in zip(row_ind, col_ind):
        if dtw_cost_matrix[r, c] < dtw_threshold:
            if verify_epipolar_constraint_points(filtered_traj_left[r], filtered_traj_right[c], F_matrix,
                                                 epipolar_thresh):
                matched_pairs_indices.append((r, c))

    print(f"经过DTW和对极约束验证后，找到 {len(matched_pairs_indices)} 对匹配的轨迹。")
    final_matched_pairs = [(filtered_traj_left_obj[r], filtered_traj_right_obj[c]) for r, c in matched_pairs_indices]
    return final_matched_pairs


def run_trajectory_matching(traj_file_left, traj_file_right, calib_params_file, output_matched_file, matching_params):
    try:
        with open(traj_file_left, 'rb') as f:
            trajectories_l = pickle.load(f)
        with open(traj_file_right, 'rb') as f:
            trajectories_r = pickle.load(f)
        calib_data = np.load(calib_params_file)
        if 'F' not in calib_data: print(f"错误: 'F'矩阵未在 {calib_params_file} 中找到。"); return
        F_matrix = calib_data['F']
    except FileNotFoundError:
        print(f"错误: 文件未找到。"); return

    matched_trajectory_pairs = match_trajectories_dtw(trajectories_l, trajectories_r, F_matrix, matching_params)
    if matched_trajectory_pairs is None: print("未找到任何匹配的轨迹对。"); return

    os.makedirs(os.path.dirname(output_matched_file), exist_ok=True)
    with open(output_matched_file, 'wb') as f:
        pickle.dump(matched_trajectory_pairs, f)
    print(f"匹配的2D轨迹对已保存至 {output_matched_file}")


if __name__ == '__main__':
    traj_l_file = "../data/trajectories/trajectories_2d_left.pkl"
    traj_r_file = "../data/trajectories/trajectories_2d_right.pkl"
    calib_file = "../camera_calibration/params/stereo_calib_params_from_matlab_full.npz"  # 确保使用全景校准文件
    out_matched_pkl = "../data/trajectories/matched_pairs_2d.pkl"

    # (已修改) 匹配参数与跟踪参数相对应
    matching_parameters = {
        'min_traj_len': 4,  # <-- 降低长度要求
        'dtw_threshold': 200.0,
        'epipolar_thresh': 2.5,
        'length_diff_ratio': 0.90
    }

    run_trajectory_matching(traj_l_file, traj_r_file, calib_file, out_matched_pkl, matching_parameters)
