# visualize_v2_11.py
#
# 这是一个为 v2.11 (16GB Hires) 模型定制的可视化脚本。
#
# 核心改动 (借鉴 v3.0 LightGlue 脚本):
# 1. 匹配训练分辨率：
#    它现在从 `sparse_reconstructor_v2_11_hires_16gb.py` 导入 Config，
#    并在 `preprocess_images` 中使用 Config 定义的高分辨率 (例如 1024x640)。
# 2. 引入坐标缩放 (v3.0 的精华):
#    在 3D 重建之前，将模型在 1024x640 分辨率下预测的 (x, y, d) 坐标，
#    按比例放大回原始的 2560x1600 空间。
#
# 这将最终解决“深度噪声”问题，让你看到高分辨率训练的真正 3D 结果。
#

import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

# --- 关键：从你的 v2.11 训练脚本中导入 ---
try:
    from sparse_reconstructor_v2_11_hires_16gb import (
        Config,
        SparseMatchingStereoModel,
        PositionalEncoding,
        SelfAttentionLayer,
        CrossAttentionLayer,
        MatchingLayer,
        SparseMatchingNetwork,
        DINOv3FeatureExtractor,
        SparseKeypointDetector
    )
except ImportError as e:
    print(f"[FATAL ERROR] 无法导入 'sparse_reconstructor_v2_11_hires_16gb.py'。")
    print(f"请确保此脚本与 'sparse_reconstructor_v2_11_hires_16gb.py' 位于同一目录。")
    print(f"详细错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[FATAL ERROR] 加载 'sparse_reconstructor_v2_11_hires_16gb.py' 时发生意外错误。")
    print(f"详细错误: {e}")
    sys.exit(1)


def preprocess_images(left_img_path, right_img_path, calib_data, cfg):
    """
    加载、校正并缩放到 *Config* 中定义的高分辨率。
    返回 tensors、可视化图像和 scale_factor。
    """

    # 1. 加载图像 (原始分辨率)
    left_raw = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_raw = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    if left_raw is None or right_raw is None:
        raise FileNotFoundError(f"无法加载图像: {left_img_path} 或 {right_img_path}")

    # 2. 加载校正数据
    map1_left, map2_left = calib_data['map1_left'], calib_data['map2_left']
    map1_right, map2_right = calib_data['map1_right'], calib_data['map2_right']
    roi_left = tuple(map(int, calib_data['roi_left']))

    # 3. 校正 (在原始分辨率下)
    left_rect = cv2.remap(left_raw, map1_left, map2_left, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_raw, map1_right, map2_right, cv2.INTER_LINEAR)

    # 4. 裁剪 (我们只使用左图的ROI)
    lx, ly, lw, lh = roi_left
    if ly + lh > left_rect.shape[0] or lx + lw > left_rect.shape[1]:
        print(f"[警告] ROI {roi_left} 超出了左图边界 {left_rect.shape}。")
        lh = left_rect.shape[0] - ly
        lw = left_rect.shape[1] - lx

    left_rect_cropped = left_rect[ly:ly + lh, lx:lx + lw]
    right_rect_cropped = right_rect[ly:ly + lh, lx:lx + lw]

    original_size = (left_rect_cropped.shape[1], left_rect_cropped.shape[0])  # (W_full, H_full)
    print(f"  原始 (裁剪后) 尺寸: {original_size[0]} x {original_size[1]}")

    # 5. 缩放 (到 v2.11 的高分辨率)
    target_size = (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)  # (W_hires, H_hires)

    left_img = cv2.resize(left_rect_cropped, target_size)
    right_img = cv2.resize(right_rect_cropped, target_size)
    print(f"  缩放到训练时尺寸: {target_size[0]} x {target_size[1]}")

    # --- v3.0 的精华：计算缩放因子 ---
    scale_w = original_size[0] / target_size[0]
    scale_h = original_size[1] / target_size[1]
    scale_factor = np.array([scale_w, scale_h])
    print(f"  计算出缩放因子: W={scale_w:.4f}, H={scale_h:.4f}")
    # ------------------------------------

    # 6. 生成 Mask
    _, mask = cv2.threshold(left_img, cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 7. 转换为 Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    left_gray = torch.from_numpy(left_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    right_gray = torch.from_numpy(right_img).float().unsqueeze(0).unsqueeze(0) / 255.0

    left_rgb = torch.from_numpy(cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float().unsqueeze(
        0) / 255.0
    right_rgb = torch.from_numpy(cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float().unsqueeze(
        0) / 255.0

    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0

    # 8. 填充以匹配 DINO patch size (16)
    b, c, h, w = left_gray.shape
    patch_size = 16
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:
        left_gray = F.pad(left_gray, (0, pad_w, 0, pad_h))
        right_gray = F.pad(right_gray, (0, pad_w, 0, pad_h))
        left_rgb = F.pad(left_rgb, (0, pad_w, 0, pad_h))
        right_rgb = F.pad(right_rgb, (0, pad_w, 0, pad_h))
        mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h))

    # 返回在 GPU 上的 tensors
    return (
        left_gray.to(device),
        right_gray.to(device),
        left_rgb.to(device),
        right_rgb.to(device),
        mask_tensor.to(device),
        left_img,  # 返回用于可视化的 Hires 图像
        scale_factor
    )


def reproject_to_3d(keypoints, disparities, Q):
    """
    使用 Q 矩阵将 (x, y, d) 转换为 (X, Y, Z)
    """
    if len(keypoints) == 0:
        return np.array([])

    num_points = keypoints.shape[0]
    points_2d_disp = np.hstack((keypoints, disparities.reshape(-1, 1)))
    points_homogeneous = np.hstack((points_2d_disp, np.ones((num_points, 1))))

    points_3d_homogeneous = (Q @ points_homogeneous.T).T

    W = points_3d_homogeneous[:, 3].reshape(-1, 1)

    W_safe = W
    W_safe[W_safe == 0] = 1.0

    points_3d = points_3d_homogeneous[:, :3] / W_safe

    return points_3d


def visualize_point_cloud(points_3d, left_image, keypoints_2d):
    """
    使用 Matplotlib 可视化 3D 点云和 2D 关键点
    (来自 v2.12)
    """

    # v2.12 修复：在绘图前设置字体，修复 2D 和 3D 的中文乱码
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 尝试使用 'SimHei' (黑体)
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except Exception as e:
        print(f"警告：设置中文字体失败: {e}。 标题可能显示为方框。")

    print(f"开始可视化... 共 {len(points_3d)} 个 3D 点。")

    fig = plt.figure(figsize=(18, 8))

    # --- 子图 1: 2D 关键点 ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(left_image, cmap='gray')
    ax1.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], c='r', s=5, alpha=0.5)
    ax1.set_title(f"2D 检测到的关键点 ({len(keypoints_2d)} 个)")
    ax1.axis('off')

    # --- 子图 2: 3D 点云 ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # --- 调试：打印原始坐标范围 ---
    if len(points_3d) > 0:
        print(f"  [调试] 原始 X 范围: {np.min(points_3d[:, 0]):.2f} to {np.max(points_3d[:, 0]):.2f}")
        print(f"  [调试] 原始 Y 范围: {np.min(points_3d[:, 1]):.2f} to {np.max(points_3d[:, 1]):.2f} (这应该是'垂直')")
        print(f"  [调试] 原始 Z 范围: {np.min(points_3d[:, 2]):.2f} to {np.max(points_3d[:, 2]):.2f} (这应该是'距离')")
    # ------------------------------------

    # --- v2.12 修复：使用百分位数自动过滤异常点 ---
    if len(points_3d) > 100:  # 确保有足够的数据来计算百分位数
        z_min_lim = np.percentile(points_3d[:, 2], 1)
        z_max_lim = np.percentile(points_3d[:, 2], 99)
        y_min_lim = np.percentile(points_3d[:, 1], 1)
        y_max_lim = np.percentile(points_3d[:, 1], 99)

        valid_mask = (points_3d[:, 2] > z_min_lim) & (points_3d[:, 2] < z_max_lim) & \
                     (points_3d[:, 1] > y_min_lim) & (points_3d[:, 1] < y_max_lim)

        points_valid = points_3d[valid_mask]
        print(f"  [调试] 自动过滤 Z 范围: [{z_min_lim:.2f}, {z_max_lim:.2f}]")
        print(f"  [调试] 自动过滤 Y 范围: [{y_min_lim:.2f}, {y_max_lim:.2f}]")
    else:
        points_valid = points_3d

    if len(points_valid) == 0:
        print("[警告] 过滤后没有找到有效的 3D 点进行可视化。")
        print("       请检查上面打印的 [调试] 原始 Z 范围。")
        ax2.set_title("3D 点云 (无有效数据)")
        plt.show()
        return

    print(f"过滤后剩余 {len(points_valid)} 个 3D 点。")

    # 绘制 3D 散点图
    # --- v2.12 核心修改：交换 Y 和 Z 轴 ---
    X_coords = points_valid[:, 0]
    Y_coords_vertical = points_valid[:, 1]  # 垂直
    Z_coords_distance = points_valid[:, 2]  # 距离
    colors = Y_coords_vertical  # 按垂直高度着色

    ax2.scatter(X_coords,  # X 轴
                Z_coords_distance,  # Y 轴 (现在是距离)
                Y_coords_vertical,  # Z 轴 (现在是垂直)
                c=colors,
                cmap='viridis_r',
                s=1)  # 点的大小

    ax2.set_title("重建的 3D 点云 (侧视图)")
    ax2.set_xlabel('X (mm - 水平)')
    ax2.set_ylabel('Z (mm - 距离)')
    ax2.set_zlabel('Y (mm - 垂直)')

    # 调整坐标轴比例
    try:
        x_min, x_max = np.min(X_coords), np.max(X_coords)
        y_min, y_max = np.min(Y_coords_vertical), np.max(Y_coords_vertical)
        z_min, z_max = np.min(Z_coords_distance), np.max(Z_coords_distance)

        range_x = x_max - x_min
        range_y = y_max - y_min
        range_z = z_max - z_min

        max_range = max(range_x, range_y, range_z)
        if max_range == 0: max_range = 1.0

        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2

        ax2.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax2.set_ylim(mid_z - max_range / 2, mid_z + max_range / 2)
        ax2.set_zlim(mid_y - max_range / 2, mid_y + max_range / 2)
    except Exception as e:
        print(f"[警告] 自动调整坐标轴比例失败: {e}")

    # v2.12 翻转 Z 轴 (现在是 Y 数据)，使其与图像坐标系匹配 (Y 向下为正)
    ax2.invert_zaxis()

    plt.tight_layout()
    plt.show()


def main():
    # --- !!! 修改这里的路径 !!! ---

    # 1. 指向你 v2.11 训练好的 .pth 文件
    #    注意：v2.11 的训练文件夹名包含了分辨率 (例如 1024x640)
    #    请确保你指向了正确的训练文件夹！
    MODEL_PATH = r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_sparse\YOUR_v2.11_TIMESTAMP_HERE\checkpoints\best_model_sparse.pth"

    # 2. 指向你的标定文件 (.npz)
    CALIBRATION_FILE_PATH = r"D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_from_matlab_full.npz"

    # 3. 指向你想要测试的一对图像
    LEFT_IMAGE_PATH = r"D:\Research\wave_reconstruction_project\data\left_images\left0001.bmp"
    RIGHT_IMAGE_PATH = r"D:\Research\wave_reconstruction_project\data\right_images\right0001.bmp"

    # --- 脚本开始 ---

    print("--- 开始 3D 可视化脚本 (v2.11.vis) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载配置 (从 v2.11 脚本)
    print("加载 v2.11 配置...")
    cfg = Config()

    # 2. 加载标定文件
    print(f"加载标定文件: {CALIBRATION_FILE_PATH}")
    try:
        calib_data = np.load(CALIBRATION_FILE_PATH)
        Q = calib_data['Q']
        print("成功加载 'Q' 矩阵。")
    except Exception as e:
        print(f"[FATAL ERROR] 加载标定文件失败: {e}")
        sys.exit(1)

    # 3. 加载 v2.11 模型
    print(f"加载模型: {MODEL_PATH}")
    try:
        model = SparseMatchingStereoModel(cfg).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()  # !!! 设置为评估模式 !!!
        print("模型加载成功。")
    except Exception as e:
        print(f"[FATAL ERROR] 加载模型失败: {e}")
        print("       请确保上面的 'MODEL_PATH' 指向了正确的 v2.11 .pth 文件！")
        sys.exit(1)

    # 4. 预处理图像 (使用 v2.11 的高分辨率)
    print("预处理图像 (使用高分辨率)...")
    try:
        left_gray, right_gray, left_rgb, right_rgb, mask, left_img_vis, scale_factor = \
            preprocess_images(LEFT_IMAGE_PATH, RIGHT_IMAGE_PATH, calib_data, cfg)
        print(f"图像已处理并发送到 {device}")
    except Exception as e:
        print(f"[FATAL ERROR] 图像预处理失败: {e}")
        sys.exit(1)

    # 5. 运行模型 (在 Hires 图像上)
    print("运行模型推断...")
    with torch.no_grad():
        try:
            outputs = model(left_gray, right_gray, left_rgb, right_rgb, mask)
        except Exception as e:
            print(f"[FATAL ERROR] 模型前向传播失败: {e}")
            print("       如果这是 OOM 错误, 也许你的 v2.11 训练使用了更低的分辨率？")
            sys.exit(1)

    # 6. 从 GPU 获取 Hires 结果
    keypoints_left_hires = outputs['keypoints_left'][0].cpu().numpy()  # (N, 2)
    scores_left = outputs['scores_left'][0].cpu().numpy()  # (N,)
    disparity_hires = outputs['disparity'][0].cpu().numpy()  # (N,)

    # 7. 过滤有效点
    valid_mask = scores_left > 0.1
    kp_valid = keypoints_left_hires[valid_mask]
    disp_valid = disparity_hires[valid_mask]

    non_zero_disp_mask = disp_valid > 1.0  # 至少 1 个像素的视差
    kp_final_hires = kp_valid[non_zero_disp_mask]
    disp_final_hires = disp_valid[non_zero_disp_mask]

    if len(kp_final_hires) == 0:
        print("[错误] 模型没有为这对图像输出任何有效的匹配点。无法生成 3D 点云。")
        sys.exit(0)

    print(f"模型输出了 {len(kp_final_hires)} 个有效匹配点 (在高分辨率 {cfg.IMAGE_WIDTH}x{cfg.IMAGE_HEIGHT} 空间)。")

    # 8. --- 关键：应用 v3.0 的缩放逻辑 ---
    # 将 (x, y) 和视差 d 放大回全分辨率空间
    print("将坐标和视差放大回全分辨率空间...")

    # (x, y) * (scale_w, scale_h)
    kp_fullres = kp_final_hires * scale_factor
    # 视差 (d) 只受宽度缩放的影响
    disp_fullres = disp_final_hires * scale_factor[0]

    print(f"  [调试] Hires 平均视差: {np.mean(disp_final_hires):.2f} 像素")
    print(f"  [调试] 映射回全分辨率的平均视差: {np.mean(disp_fullres):.2f} 像素")

    # 9. 3D 重建 (使用全分辨率坐标)
    print("正在使用全分辨率坐标重建 3D 点云...")
    points_3d = reproject_to_3d(kp_fullres, disp_fullres, Q)

    # 10. 可视化
    visualize_point_cloud(points_3d, left_img_vis, kp_final_hires)


if __name__ == "__main__":
    main()