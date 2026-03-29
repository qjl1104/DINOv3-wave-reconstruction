# visualize_3d.py
# 加载训练好的模型，对输入的双目图像进行推理，生成视差图，并重建为可交互的3D点云图。

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import plotly.graph_objects as go
from tqdm import tqdm

# 导入主训练脚本中的模型和配置类
# 这确保了模型加载时架构的一致性
from train_stereo_model import Config, DINOv3StereoModel


def main(args):
    """主执行函数"""
    print("--- 开始三维重建与可视化 ---")

    # --- 1. 加载配置和标定文件 ---
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 使用设备: {device}")

    try:
        calib = np.load(cfg.CALIBRATION_FILE)
        Q = calib['Q']
        map1_left, map2_left = calib['map1_left'], calib['map2_left']
        map1_right, map2_right = calib['map1_right'], calib['map2_right']
        roi_left = tuple(calib['roi_left'])
        print(f"✓ 成功加载相机标定文件: {cfg.CALIBRATION_FILE}")
    except Exception as e:
        print(f"【致命错误】加载标定文件 '{cfg.CALIBRATION_FILE}' 失败: {e}")
        sys.exit(1)

    # --- 2. 加载模型 ---
    print(f"正在加载模型检查点: {args.checkpoint}")
    model = DINOv3StereoModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("✓ 模型加载成功")

    # --- 3. 加载并处理图像 ---
    print(f"正在加载图像: \n  左图: {args.left_image}\n  右图: {args.right_image}")
    left_img_raw = cv2.imread(args.left_image, cv2.IMREAD_GRAYSCALE)
    right_img_raw = cv2.imread(args.right_image, cv2.IMREAD_GRAYSCALE)
    if left_img_raw is None or right_img_raw is None:
        print("【致命错误】无法读取一个或两个输入图像。")
        sys.exit(1)

    # 校正图像
    left_rectified = cv2.remap(left_img_raw, map1_left, map2_left, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img_raw, map1_right, map2_right, cv2.INTER_LINEAR)

    # 裁剪到ROI
    x, y, w, h = roi_left
    left_rectified_cropped = left_rectified[y:y + h, x:x + w]
    original_height, original_width = left_rectified_cropped.shape[:2]

    # 预处理以匹配模型输入
    target_h, target_w = cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH
    left_img_resized = cv2.resize(left_rectified_cropped, (target_w, target_h))
    right_img_resized = cv2.resize(right_rectified[y:y + h, x:x + w], (target_w, target_h))

    left_img_rgb = cv2.cvtColor(left_img_resized, cv2.COLOR_GRAY2BGR)
    right_img_rgb = cv2.cvtColor(right_img_resized, cv2.COLOR_GRAY2BGR)

    left_tensor = torch.from_numpy(left_img_rgb.transpose(2, 0, 1)).float().to(device) / 255.0
    right_tensor = torch.from_numpy(right_img_rgb.transpose(2, 0, 1)).float().to(device) / 255.0
    left_tensor = left_tensor.unsqueeze(0)
    right_tensor = right_tensor.unsqueeze(0)
    print("✓ 图像预处理完成")

    # --- 4. 模型推理 ---
    print("正在进行模型推理以生成视差图...")
    with torch.no_grad():
        outputs = model(left_tensor, right_tensor)
        disparity_map_tensor = outputs["disparity"].squeeze(0).squeeze(0)

    disparity_map_resized = disparity_map_tensor.cpu().numpy()

    # 将视差图恢复到原始ROI尺寸
    disparity_map = cv2.resize(disparity_map_resized, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    print("✓ 视差图生成成功")

    # --- 5. 三维重建 ---
    print("正在使用Q矩阵将视差图重投影到3D空间...")

    # 过滤掉无效的视差值
    mask = disparity_map > args.min_disparity
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)

    # 获取颜色信息
    colors = cv2.cvtColor(left_rectified_cropped, cv2.COLOR_GRAY2BGR)

    # 应用掩码
    points_3d = points_3d[mask]
    colors = colors[mask]

    # 进一步过滤掉距离过远或无效的点
    z_coords = points_3d[:, 2]
    valid_points_mask = (z_coords < args.max_depth) & (z_coords > 0)
    points_3d = points_3d[valid_points_mask]
    colors = colors[valid_points_mask]

    num_points = points_3d.shape[0]
    if num_points == 0:
        print("【警告】没有有效的3D点被重建。可能是视差图为空或所有点都被过滤掉了。")
        return

    print(f"✓ 成功重建 {num_points} 个有效3D点")

    # --- 6. Plotly交互式可视化 ---
    print("正在生成交互式3D点云图...")
    fig = go.Figure(data=[go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='markers',
        marker=dict(
            size=args.point_size,
            color=colors.reshape(-1, 3)[:, ::-1],  # BGR to RGB
            opacity=0.8
        )
    )])

    fig.update_layout(
        title='交互式3D点云重建结果',
        scene=dict(
            xaxis_title='X (毫米)',
            yaxis_title='Y (毫米)',
            zaxis_title='Z (深度/毫米)',
            aspectmode='data'  # 保持xyz轴的比例
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )

    output_filename = os.path.basename(args.left_image).split('.')[0] + "_3d_reconstruction.html"
    fig.write_html(output_filename)
    print(f"✓ 可视化结果已保存至: {output_filename}")
    print("--- 任务完成 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从双目图像生成3D点云')
    parser.add_argument('--left_image', type=str,
                        default=r"D:\Research\wave_reconstruction_project\data\left_images\left0001.bmp",
                        help='左相机原始图像的路径. 默认为示例路径.')
    parser.add_argument('--right_image', type=str,
                        default=r"D:\Research\wave_reconstruction_project\data\right_images\right0001.bmp",
                        help='右相机原始图像的路径. 默认为示例路径.')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints_self_supervised/best_model_self_supervised.pth',
                        help='训练好的模型检查点 (.pth) 文件的路径.')
    parser.add_argument('--min_disparity', type=float, default=1.0, help='用于过滤背景的最小视差值')
    parser.add_argument('--max_depth', type=float, default=10000.0, help='重建点的最大深度值 (Z坐标)')
    parser.add_argument('--point_size', type=int, default=2, help='3D图中点的大小')

    args = parser.parse_args()

    # 在执行主函数前，检查文件是否存在
    if not os.path.exists(args.left_image):
        print(f"【错误】找不到左图文件: '{args.left_image}'")
        sys.exit(1)
    if not os.path.exists(args.right_image):
        print(f"【错误】找不到右图文件: '{args.right_image}'")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"【错误】找不到模型检查点文件: '{args.checkpoint}'")
        print("请确认路径是否正确，或先运行 train_stereo_model.py 完成训练。")
        sys.exit(1)

    main(args)

