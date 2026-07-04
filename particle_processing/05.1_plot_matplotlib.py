# 05.1_plot_matplotlib.py
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 1. 添加这两行代码来解决中文字符显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定一个支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示不正常的问题
# --------------------------------------------------

# 2. 定义你的文件路径
output_3d_traj_file = r"D:\Research\wave_reconstruction_project\data/trajectories/trajectories_3d.pkl"

# 3. 加载 3D 轨迹数据
try:
    with open(output_3d_traj_file, 'rb') as f:
        all_3d_trajectories = pickle.load(f)
    print(f"成功加载 {len(all_3d_trajectories)} 条 3D 轨迹。")
except FileNotFoundError:
    print(f"错误: 未找到文件 {output_3d_traj_file}")
    exit()

# 4. 创建 3D 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 5. 绘制每一条轨迹
for i, traj_3d in enumerate(all_3d_trajectories):
    # 将列表转换为 NumPy 数组
    traj_3d_np = np.array(traj_3d)

    # 提取 x, y, z 坐标
    x = traj_3d_np[:, 0]
    y = traj_3d_np[:, 1]
    z = traj_3d_np[:, 2]

    # 绘制轨迹线
    ax.plot(x, y, z, label=f'轨迹 {i + 1}')

    # 绘制起点和终点
    ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=20)  # 起点
    ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=20)  # 终点

# 6. 设置图表属性
ax.set_xlabel('X 坐标')
ax.set_ylabel('Y 坐标')
ax.set_zlabel('Z 坐标')
ax.set_title('3D 粒子轨迹可视化')
ax.grid(True)

# 7. 显示图表
plt.show()