# wave_modeling/08_training.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from model_definition import PINN_Wave # 假设模型定义在同目录下
# from data_preparation import prepare_pinn_data_from_trajectories # 假设数据准备在同目录下
import os
import numpy as np # for collocation points if not loaded from file

def train_pinn_model(
    trajectories_3d_file, 
    model_save_path,
    epochs=10000, 
    lr=1e-3, 
    lambda_physics=1e-2, 
    g_const=9.81, 
    water_depth_h=1.0, # 示例水深
    batch_size_data=1024,
    num_collocation_points=20000
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 数据准备
    particle_dataset, collocation_xyt = prepare_pinn_data_from_trajectories(trajectories_3d_file, num_collocation_points)
    if particle_dataset is None or collocation_xyt is None:
        print("Failed to prepare data. Exiting training.")
        return

    particle_loader = DataLoader(particle_dataset, batch_size=batch_size_data, shuffle=True, drop_last=True)
    # Collocation points are typically fixed for one training run or sampled once per epoch
    collocation_xyt = collocation_xyt.to(device)
    collocation_xyt.requires_grad_(True) # Crucial for autograd

    # 2. 模型和优化器
    model = PINN_Wave().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting PINN training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss_data_sum = 0
        epoch_loss_physics_sum = 0
        num_batches = 0

        for xyt_particle_batch, z_observed_batch in particle_loader:
            xyt_particle_batch = xyt_particle_batch.to(device)
            z_observed_batch = z_observed_batch.to(device)

            optimizer.zero_grad()

            # 数据拟合损失
            eta_predicted_particle = model(xyt_particle_batch)
            loss_data = torch.mean((eta_predicted_particle - z_observed_batch)**2)

            # 物理残差损失
            # (可选) 可以在每个epoch或每个batch重新采样配置点
            # collocation_xyt.requires_grad_(True) # 确保每次迭代前都设置
            pde_residual = model.physics_residual(collocation_xyt, g_const, water_depth_h)
            loss_physics = torch.mean(pde_residual**2)

            total_loss = loss_data + lambda_physics * loss_physics

            total_loss.backward() # 反向传播
            optimizer.step()

            epoch_loss_data_sum += loss_data.item()
            epoch_loss_physics_sum += loss_physics.item()
            num_batches += 1

        avg_loss_data = epoch_loss_data_sum / num_batches
        avg_loss_physics = epoch_loss_physics_sum / num_batches

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss_data: {avg_loss_data:.6e}, Avg Loss_physics: {avg_loss_physics:.6e}, Total: {(avg_loss_data + lambda_physics * avg_loss_physics):.6e}")

    # 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained PINN model saved to {model_save_path}")

if __name__ == '__main__':
    # 确保这些辅助脚本在同一目录或Python路径中
    from model_definition import PINN_Wave 
    from data_preparation import prepare_pinn_data_from_trajectories

    traj_file = "../../data/trajectories/trajectories_3d.pkl"
    save_model_to = "saved_models/pinn_wave_v1.pth"

    train_pinn_model(
        traj_file, 
        save_model_to,
        epochs=2000, # 示例 epochs
        lr=0.001,
        lambda_physics=0.01 
    )