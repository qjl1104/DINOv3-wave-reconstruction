# particle_processing/03_trajectory_tracking_2d.py
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
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

    def predict(self): return self.kf.predict()

    def update(self, detection, frame_idx): self.kf.update(np.array(detection)); self.points[
        frame_idx] = detection; self.age = 0; self.total_visible_count += 1; self.last_frame_seen = frame_idx

    def get_ordered_points(self): return [self.points[fi] for fi in sorted(self.points.keys())]

    def mark_missed(self): self.age += 1

    def is_tentative(self, min_hits_for_confirmation=3): return self.total_visible_count < min_hits_for_confirmation

    def is_lost(self, max_age_limit=5): return self.age > max_age_limit


def track_particles_kalman_hungarian(all_detections_per_frame, max_age=5, min_hits_to_confirm=3, dist_thresh=50.0):
    active_tracks, completed_tracks, next_track_id = [], [], 0
    total_frames = len(all_detections_per_frame)
    for frame_idx, detections_current_frame in enumerate(all_detections_per_frame):
        predicted_positions = [track.predict() for track in active_tracks]
        num_active_tracks, num_detections = len(active_tracks), len(detections_current_frame)
        cost_matrix = np.full((num_active_tracks, num_detections), np.inf)
        if num_active_tracks > 0 and num_detections > 0:
            for t_idx, pred_pos in enumerate(predicted_positions):
                for d_idx, det_pos in enumerate(detections_current_frame):
                    dist = np.linalg.norm(np.array(pred_pos) - np.array(det_pos))
                    if dist < dist_thresh: cost_matrix[t_idx, d_idx] = dist

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

        new_active_tracks = []
        for i, track in enumerate(active_tracks):
            if i not in matched_track_indices_set:
                track.mark_missed()
                if track.is_lost(max_age):
                    if not track.is_tentative(min_hits_to_confirm): completed_tracks.append(track)
                else:
                    new_active_tracks.append(track)
            else:
                new_active_tracks.append(track)
        active_tracks = new_active_tracks

        for i, det in enumerate(detections_current_frame):
            if i not in matched_detection_indices_set:
                new_track = Track(next_track_id, det, frame_idx);
                active_tracks.append(new_track);
                next_track_id += 1

        if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == total_frames:
            print(f"已处理 {frame_idx + 1}/{total_frames} 帧。当前活跃轨迹: {len(active_tracks)}")

    for track in active_tracks:
        if not track.is_tentative(min_hits_to_confirm): completed_tracks.append(track)
    return [t for t in completed_tracks if len(t.points) >= min_hits_to_confirm]


def run_tracking(detections_file_left, detections_file_right, output_traj_left, output_traj_right, params_left,
                 params_right):
    try:
        with open(detections_file_left, 'rb') as f:
            all_detections_left = pickle.load(f)
        with open(detections_file_right, 'rb') as f:
            all_detections_right = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 检测文件未找到。"); return

    print("\n--- 开始左侧相机的2D轨迹跟踪 ---")
    trajectories_left = track_particles_kalman_hungarian(all_detections_left, **params_left)
    print(f"为左侧相机找到 {len(trajectories_left)} 条轨迹。")

    print("\n--- 开始右侧相机的2D轨迹跟踪 ---")
    trajectories_right = track_particles_kalman_hungarian(all_detections_right, **params_right)
    print(f"为右侧相机找到 {len(trajectories_right)} 条轨迹。")

    os.makedirs(os.path.dirname(output_traj_left), exist_ok=True)
    with open(output_traj_left, 'wb') as f:
        pickle.dump(trajectories_left, f)
    print(f"\n左侧2D轨迹已保存至 {output_traj_left}")

    os.makedirs(os.path.dirname(output_traj_right), exist_ok=True)
    with open(output_traj_right, 'wb') as f:
        pickle.dump(trajectories_right, f)
    print(f"右侧2D轨迹已保存至 {output_traj_right}")


if __name__ == '__main__':
    det_left_file = "../data/detections/detections_left.pkl"
    det_right_file = "../data/detections/detections_right.pkl"
    out_traj_l_file = "../data/trajectories/trajectories_2d_left.pkl"
    out_traj_r_file = "../data/trajectories/trajectories_2d_right.pkl"

    # (已修改) 放宽跟踪参数，特别是 min_hits_to_confirm
    tracking_params = {
        'max_age': 5,
        'min_hits_to_confirm': 3,  # <-- 从5降低到3，更容易形成轨迹
        'dist_thresh': 50.0
    }

    # 为左右相机使用相同的宽松参数
    run_tracking(
        det_left_file, det_right_file,
        out_traj_l_file, out_traj_r_file,
        tracking_params, tracking_params
    )
