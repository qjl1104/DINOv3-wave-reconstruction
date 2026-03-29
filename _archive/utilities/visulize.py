# visualize_3d_with_diagnostics.py
import torch
import cv2
import numpy as np
import open3d as o3d
import sys
import os

# --- 关键步骤：从你的训练脚本中导入模型和配置类 ---
# 确保此脚本与 dinov3_deepseek_2.py 在同一目录下
try:
    # 根据你之前提供的信息，你的训练脚本名为 dinov3_deepseek_2.py
    from dinov3_wave_stereo2 import Config, EnhancedDINOv3StereoModel
except ImportError:
    print("=" * 80)
    print("【错误】: 无法找到 'dinov3_deepseek_2.py'。")
    print("请确保此脚本与你的原始训练脚本在同一个文件夹中，")
    print("并且原始训练脚本文件名中的 '-' 已被替换为 '_'。")
    print("=" * 80)
    sys.exit(1)

# --- 1. 参数配置 (请根据你的实际情况修改) ---

# 指向你训练好的模型权重文件
MODEL_WEIGHTS_PATH = "./checkpoints_self_supervised/best_model_self_supervised.pth"

# 指向你想进行三维重建的一对图片
# 替换成你自己的图片路径！
LEFT_IMAGE_PATH = "D:/Research/wave_reconstruction_project/data/lresult/lresult0001.bmp"
RIGHT_IMAGE_PATH = "D:/Research/wave_reconstruction_project/data/rresult/rresult0001.bmp"

# 你的相机标定文件路径 (与训练脚本中保持一致)
CALIBRATION_FILE_PATH = "D:/Research/wave_reconstruction_project/camera_calibration/params/stereo_calib_params_from_matlab_full.npz"

# 过滤掉视差值过小的点，这些点通常是噪声或天空
MIN_DISPARITY_THRESHOLD = 5.0


def preprocess_image(image_path, cfg):
    """加载并预处理图像，使其符合模型输入要求"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    img_resized = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    return img_tensor.unsqueeze(0), img_resized


def visualize_point_cloud(points, colors):
    """使用Open3D可视化点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\n--- Open3D 可视化窗口操作指南 ---")
    print("  - 鼠标左键 + 拖动: 旋转视角")
    # ... (其他指南)
    o3d.visualization.draw_geometries([pcd])


def main():
    """主执行函数"""
    # --- 步骤 1: 加载配置和模型 ---
    print("步骤 1/5: 正在加载配置和模型...")
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EnhancedDINOv3StereoModel(cfg).to(device)
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"【致命错误】: 找不到模型权重文件 '{MODEL_WEIGHTS_PATH}'。")
        print("请确认路径是否正确，以及模型是否已训练完成。")
        sys.exit(1)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"✓ 模型加载成功，使用设备: {device}")

    # --- 步骤 2: 加载并预处理图像 ---
    print("步骤 2/5: 正在加载并预处理图像...")
    left_tensor, left_img_for_color = preprocess_image(LEFT_IMAGE_PATH, cfg)
    right_tensor, _ = preprocess_image(RIGHT_IMAGE_PATH, cfg)
    left_tensor, right_tensor = left_tensor.to(device), right_tensor.to(device)
    print(f"✓ 图像加载成功: {os.path.basename(LEFT_IMAGE_PATH)}")

    # --- 步骤 3: 模型推理，获取视差图 ---
    print("步骤 3/5: 正在进行模型推理以获取视差图...")
    with torch.no_grad():
        outputs = model(left_tensor, right_tensor)
        disparity_tensor = outputs["disparity"]

    disparity_np = disparity_tensor.squeeze().cpu().numpy()
    print("✓ 视差图预测完成。")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++【诊断代码修改】+++
    # +++ 我们在这里打印出模型输出的原始数值，以诊断“一片蓝”的问题。
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    min_val = disparity_np.min()
    max_val = disparity_np.max()
    mean_val = disparity_np.mean()
    std_val = disparity_np.std()

    print("\n" + "=" * 25 + " 诊断信息 " + "=" * 25)
    print(f"  视差图统计: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}, Std Dev={std_val:.4f}")

    # 根据诊断信息给出判断
    if max_val < 1.0 and std_val < 0.1:
        print("  [诊断结论]: 视差值范围非常小且几乎没有变化 (Std Dev很低)。")
        print("  这强烈表明模型没有被充分训练，或者加载了错误的权重。")
        print("  因此，视差图看起来会是纯色（例如一片蓝）。")
    else:
        print("  [诊断结论]: 视差值看起来有一定范围和变化，模型可能已学到一些特征。")
    print("=" * 64 + "\n")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # (可选) 保存视差图为图片，便于检查
    # 增加一个 epsilon 防止除以零
    disp_vis_normalized = (disparity_np - min_val) / (max_val - min_val + 1e-8)
    disp_vis_u8 = (disp_vis_normalized * 255.0).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis_u8, cv2.COLORMAP_JET)
    cv2.imwrite("disparity_map_output.png", disp_color)
    print("   - 提示: 预测的彩色视差图已保存为 'disparity_map_output.png'")

    # --- 步骤 4: 加载Q矩阵并重建三维点云 ---
    print("步骤 4/5: 正在加载相机标定文件并重建三维点云...")
    try:
        calib_data = np.load(CALIBRATION_FILE_PATH)
        if 'Q' not in calib_data:
            raise KeyError("标定文件中找不到名为 'Q' 的矩阵。")
        Q = calib_data['Q']
    except (FileNotFoundError, KeyError) as e:
        print(f"【致命错误】: 加载标定文件失败: {e}")
        sys.exit(1)

    points_3d = cv2.reprojectImageTo3D(disparity_np, Q)
    print("✓ 三维点云重建完成。")

    # --- 步骤 5: 格式化点云数据并进行可视化 ---
    print("步骤 5/5: 正在准备数据并启动可视化...")
    colors = cv2.cvtColor(left_img_for_color, cv2.COLOR_BGR2RGB)
    mask = disparity_np > MIN_DISPARITY_THRESHOLD
    points = points_3d[mask]
    colors = colors[mask]

    if len(points) == 0:
        print("\n【警告】: 没有满足视差阈值的有效三维点可供显示。")
        print(f"   - 当前阈值 (MIN_DISPARITY_THRESHOLD) 为: {MIN_DISPARITY_THRESHOLD}")
        print(f"   - 而你的模型预测的最大视差值为: {max_val:.4f}")
        print("   - 尝试在脚本开头降低阈值，或者检查模型训练情况。")
        return

    colors = colors.reshape(-1, 3) / 255.0
    visualize_point_cloud(points, colors)
    print("\n可视化结束。")


if __name__ == '__main__':
    main()
