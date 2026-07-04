# wave_modeling/06_data_preparation.py
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class ParticlePINNDataset(Dataset):
    def __init__(self, particle_points_xyz, time_indices):
        # particle_points_xyz: list of [x, y, z]
        # time_indices: list of corresponding time indices/values
        self.xyt = torch.tensor([[p, p[1], t] for p, t in zip(particle_points_xyz, time_indices)], dtype=torch.float32)
        self.z_observed = torch.tensor([[p[1]] for p in particle_points_xyz], dtype=torch.float32)

    def __len__(self):
        return len(self.z_observed)

    def __getitem__(self, idx):
        return self.xyt[idx], self.z_observed[idx]

def prepare_pinn_data_from_trajectories(trajectories_3d_file, num_collocation_points=10000):
    try:
        with open(trajectories_3d_file, 'rb') as f:
            all_3d_trajectories = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: 3D trajectories file {trajectories_3d_file} not found.")
        return None, None

    particle_points_xyz =
    time_indices_for_particles =

    # 假设每个轨迹中的点对应一个时间步（帧索引）
    # 您可能需要更复杂的时间戳管理
    for traj in all_3d_trajectories:
        for frame_idx, point_3d in enumerate(traj):
            particle_points_xyz.append(point_3d) # [x,y,z]
            time_indices_for_particles.append(float(frame_idx)) # 使用帧索引作为时间

    if not particle_points_xyz:
        print("No 3D particle points found to prepare data.")
        return None, None

    # 创建Dataset和DataLoader
    particle_dataset = ParticlePINNDataset(particle_points_xyz, time_indices_for_particles)

    # 生成配置点 (Collocation points)
    # 需要根据您的数据范围来确定 x_min, x_max, y_min, y_max, t_min, t_max
    # 这里用占位符
    x_min, x_max = np.min(np.array(particle_points_xyz)[:,0]), np.max(np.array(particle_points_xyz)[:,0])
    y_min, y_max = np.min(np.array(particle_points_xyz)[:,1]), np.max(np.array(particle_points_xyz)[:,1])
    t_min, t_max = np.min(time_indices_for_particles), np.max(time_indices_for_particles)

    # 确保范围有效
    if not all(np.isfinite([x_min, x_max, y_min, y_max, t_min, t_max])):
         print("Warning: Could not determine valid bounds for collocation points. Using defaults.")
         x_min, x_max, y_min, y_max, t_min, t_max = -1, 1, -1, 1, 0, 100


    coll_x = torch.rand(num_collocation_points, 1) * (x_max - x_min) + x_min
    coll_y = torch.rand(num_collocation_points, 1) * (y_max - y_min) + y_min
    coll_t = torch.rand(num_collocation_points, 1) * (t_max - t_min) + t_min
    collocation_xyt = torch.cat([coll_x, coll_y, coll_t], dim=1).float()

    return particle_dataset, collocation_xyt

if __name__ == '__main__':
    traj_3d_file = "../../data/trajectories/trajectories_3d.pkl"
    p_dataset, c_points = prepare_pinn_data_from_trajectories(traj_3d_file)
    if p_dataset and c_points is not None:
        print(f"Prepared particle dataset with {len(p_dataset)} points.")
        print(f"Prepared {len(c_points)} collocation points.")
        # 可以进一步创建DataLoader
        # particle_loader = DataLoader(p_dataset, batch_size=1024, shuffle=True)
        # collocation_loader = DataLoader(TensorDataset(c_points), batch_size=num_collocation_points, shuffle=False) # 通常配置点一次性加载