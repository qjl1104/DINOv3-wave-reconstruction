# particle_processing/05_reconstruction_3d.py
import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob


# ----------------- 兼容性类定义 -----------------
# 这些类确保能够加载之前保存的轨迹文件
class SimpleKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 1e-1
        self.Q = np.eye(4) * 1e-3
        self.R = np.eye(2) * 1e-1
        self.dt = 1.0


class ImprovedKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0, 0.0], dtype=float)


class ExtendedKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.1, 0.0], dtype=float)


class OptimizedExtendedKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)


class UltraOptimizedKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)


class WaveParticleKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)


class StrictWaveKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)


class Track:
    def __init__(self, track_id, initial_detection, frame_idx):
        self.id = track_id
        self.points = {frame_idx: initial_detection}
        self.age = 0
        self.total_visible_count = 1
        self.last_frame_seen = frame_idx

    def get_ordered_points(self):
        return [self.points[fi] for fi in sorted(self.points.keys())]


class UltraTrack(Track):
    pass


class WaveParticleTrack(Track):
    pass


class StrictTrack(Track):
    pass


# ------------------------------------------------

def reconstruct_3d_from_matched_pair(matched_traj_pair, P1, P2):
    traj_left_2d, traj_right_2d = matched_traj_pair

    # 确保轨迹不为空，并且有相同的长度
    num_points = min(len(traj_left_2d.points), len(traj_right_2d.points))
    if num_points == 0:
        return None

    # 从类实例中获取点
    points_l_np = np.array(traj_left_2d.get_ordered_points()[:num_points], dtype=np.float32).T
    points_r_np = np.array(traj_right_2d.get_ordered_points()[:num_points], dtype=np.float32).T

    points_4d_hom = cv2.triangulatePoints(P1, P2, points_l_np, points_r_np)

    # 转换为非齐次坐标
    points_3d_non_hom = points_4d_hom[:3] / (points_4d_hom[3] + 1e-6)

    trajectory_3d = points_3d_non_hom.T.tolist()
    return trajectory_3d


def visualize_3d_trajectories(all_3d_trajectories, output_file):
    """使用 Matplotlib 绘制 3D 轨迹图"""
    # 解决中文字符显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    print(f"\n正在生成 {len(all_3d_trajectories)} 条 3D 轨迹的可视化图...")
    for i, traj_3d in enumerate(all_3d_trajectories):
        if len(traj_3d) > 1:
            traj_3d_np = np.array(traj_3d)

            x = traj_3d_np[:, 0]
            y = traj_3d_np[:, 1]
            z = traj_3d_np[:, 2]

            # 绘制轨迹线
            ax.plot(x, y, z, alpha=0.7, label=f'Trajectory {i + 1}')

            # 绘制起点和终点
            ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=20)
            ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=20)

    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_zlabel('Z 坐标')
    ax.set_title('3D 粒子轨迹重建结果')
    ax.grid(True)

    # 保存图像
    vis_path = output_file.replace('.pkl', '_vis.png')
    plt.savefig(vis_path, dpi=300)
    print(f"3D 轨迹可视化图像已保存至: {vis_path}")

    plt.show()


def run_3d_reconstruction(matched_pairs_file, calib_params_file, output_3d_traj_file, visualize=True):
    try:
        with open(matched_pairs_file, 'rb') as f:
            matched_trajectory_pairs = pickle.load(f)
        calib_data = np.load(calib_params_file)
        P1 = calib_data['P1']
        P2 = calib_data['P2']
    except FileNotFoundError:
        print(
            f"Error: Matched pairs or calibration file not found for 3D reconstruction ({matched_pairs_file} or {calib_params_file}).")
        return
    except KeyError as e:
        print(f"Error: Missing projection matrix {e} in calibration file.")
        return

    all_3d_trajectories = []
    print(f"Reconstructing 3D trajectories for {len(matched_trajectory_pairs)} matched pairs...")
    for i, pair in enumerate(matched_trajectory_pairs):
        traj_3d = reconstruct_3d_from_matched_pair(pair, P1, P2)
        if traj_3d:
            all_3d_trajectories.append(traj_3d)
        if (i + 1) % 100 == 0: print(f"  Reconstructed {i + 1}/{len(matched_trajectory_pairs)} trajectories.")
    print(f"Completed reconstruction of {len(all_3d_trajectories)} trajectories.")

    os.makedirs(os.path.dirname(output_3d_traj_file), exist_ok=True)
    with open(output_3d_traj_file, 'wb') as f:
        pickle.dump(all_3d_trajectories, f)
    print(f"All 3D trajectories saved to {output_3d_traj_file}")

    if visualize:
        visualize_3d_trajectories(all_3d_trajectories, output_3d_traj_file)


if __name__ == '__main__':
    # 文件路径 - 参照 04_trajectory_matching_optimized.py
    matched_file = "../data/trajectories/matched_pairs_2d_optimized.pkl"
    calib_file = "../camera_calibration/params/stereo_calib_params_from_matlab_full.npz"
    out_3d_file = "../data/trajectories/trajectories_3d.pkl"

    run_3d_reconstruction(matched_file, calib_file, out_3d_file)