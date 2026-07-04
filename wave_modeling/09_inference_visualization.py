# wave_modeling/09_inference_visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from model_definition import PINN_Wave # 假设模型定义在同目录下
import os

def infer_and_visualize_wave_surface(
    model_path, 
    time_point_to_plot, 
    x_range=(-10, 10), 
    y_range=(-10, 10), 
    grid_density=100
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    # 确保模型类定义可用
    from model_definition import PINN_Wave # 需要能够访问模型类
    model = PINN_Wave().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()

    # 创建空间网格
    x_coords = np.linspace(x_range, x_range[1], grid_density)
    y_coords = np.linspace(y_range, y_range[1], grid_density)
    X, Y = np.meshgrid(x_coords, y_coords)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.full_like(x_flat, float(time_point_to_plot)) # 时间点

    # 准备输入张量
    query_xyt = torch.tensor(np.vstack([x_flat, y_flat, t_flat]).T, dtype=torch.float32).to(device)

    predicted_eta_flat = None
    with torch.no_grad():
        predicted_eta_flat = model(query_xyt).cpu().numpy()

    predicted_eta_grid = predicted_eta_flat.reshape(X.shape)

    # 可视化
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, predicted_eta_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Wave Height (eta)')
    ax.set_title(f"Reconstructed Wave Surface at t = {time_point_to_plot}")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if __name__ == '__main__':
    # 确保模型定义可用
    from model_definition import PINN_Wave

    trained_model_path = "saved_models/pinn_wave_v1.pth" # 训练好的模型路径
    time_to_plot = 50.0  # 选择一个时间点进行可视化 (需要与训练时的时间尺度一致)

    # 定义绘图的空间范围，应根据您的数据调整
    plot_x_range = (-5, 5) 
    plot_y_range = (-5, 5)

    if not os.path.exists(trained_model_path):
        print(f"Model file {trained_model_path} does not exist. Please train the model first.")
    else:
        infer_and_visualize_wave_surface(trained_model_path, time_to_plot, plot_x_range, plot_y_range)