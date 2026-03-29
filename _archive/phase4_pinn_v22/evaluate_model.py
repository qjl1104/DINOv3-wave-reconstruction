# evaluate_model.py
# 描述: 该脚本用于加载训练好的稀疏匹配模型，
# 对测试图像进行推理，重建三维点云，并对结果进行可视化。
# 新功能: 评估完成后会自动用浏览器打开生成的三维点云图。

import os
import sys
import glob
import json
import dataclasses
import webbrowser
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- 中文显示配置 ---
# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# --- 配置结束 ---

# 导入训练脚本中的必要类
try:
    from sparse_reconstructor_1013_gemini import Config, SparseMatchingStereoModel
except ImportError:
    print("错误: 无法导入 'sparse_reconstructor_1013_gemini.py'。")
    print("请确保此评估脚本与您的训练脚本在同一个文件夹中。")
    sys.exit(1)


class Evaluator:
    """负责加载模型、处理数据、执行推理和可视化的类"""

    def __init__(self, run_dir: str):
        """
        初始化 Evaluator
        参数:
            run_dir (str): 训练运行时生成的目录路径，包含 checkpoints 和 logs。
        """
        self.run_dir = run_dir
        self.log_file = os.path.join(self.run_dir, "logs", "training_log.json")
        self.model_path = os.path.join(self.run_dir, "checkpoints", "best_model_sparse.pth")
        self.output_dir = os.path.join(self.run_dir, "evaluation_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载训练配置
        self.cfg = self._load_config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载相机标定参数
        try:
            calib = np.load(self.cfg.CALIBRATION_FILE)
            self.Q = calib['Q']
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            self.roi_left, self.roi_right = tuple(calib['roi_left']), tuple(calib['roi_right'])

        except Exception as e:
            sys.exit(f"加载相机标定文件失败: {e}")

        # 初始化模型并加载权重
        self.model = SparseMatchingStereoModel(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"模型已从 '{self.model_path}' 加载。")

        # 获取图像文件列表
        self.left_images = sorted(glob.glob(os.path.join(self.cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"在 '{self.cfg.LEFT_IMAGE_DIR}' 中未找到图像。")

    def _load_config(self) -> Config:
        """从日志文件中加载训练配置"""
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            config_dict = log_data['config']
            config_fields = {f.name for f in dataclasses.fields(Config)}
            filtered_config_dict = {k: v for k, v in config_dict.items() if k in config_fields}
            return Config(**filtered_config_dict)
        except Exception as e:
            sys.exit(f"加载配置文件失败: {e}")

    def load_and_preprocess_image(self, image_index: int):
        """加载并预处理一对立体图像，与训练时保持一致"""
        left_path = self.left_images[image_index]
        right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, 'right' + os.path.basename(left_path)[4:])

        left_raw = cv2.imread(left_path, 0)
        right_raw = cv2.imread(right_path, 0)

        if left_raw is None or right_raw is None:
            raise IOError(f"无法读取图像对: {left_path}, {right_path}")

        left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

        x, y, w, h = self.roi_left
        left_rect_cropped = left_rect[y:y + h, x:x + w]
        x, y, w, h = self.roi_right
        right_rect_cropped = right_rect[y:y + h, x:x + w]

        left_img_resized = cv2.resize(left_rect_cropped, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
        right_img_resized = cv2.resize(right_rect_cropped, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))

        left_gray_tensor = torch.from_numpy(left_img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        right_gray_tensor = torch.from_numpy(right_img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        left_rgb_tensor = torch.from_numpy(
            cv2.cvtColor(left_img_resized, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        right_rgb_tensor = torch.from_numpy(
            cv2.cvtColor(right_img_resized, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        return {
            'display_left': left_img_resized,
            'display_right': right_img_resized,
            'left_gray': left_gray_tensor,
            'right_gray': right_gray_tensor,
            'left_rgb': left_rgb_tensor,
            'right_rgb': right_rgb_tensor
        }

    @torch.no_grad()
    def run_inference(self, image_tensors):
        """在单对图像上运行模型推理"""
        left_gray = image_tensors['left_gray'].to(self.device)
        right_gray = image_tensors['right_gray'].to(self.device)
        left_rgb = image_tensors['left_rgb'].to(self.device)
        right_rgb = image_tensors['right_rgb'].to(self.device)
        mask = torch.ones_like(left_gray)
        outputs = self.model(left_gray, right_gray, left_rgb, right_rgb, mask)
        kp_left = outputs['keypoints_left'][0].cpu().numpy()
        scores_left = outputs['scores_left'][0].cpu().numpy()
        disparity = outputs['disparity'][0].cpu().numpy()
        valid_mask = scores_left > 0.1
        return {
            'kp_left': kp_left[valid_mask],
            'disparity': disparity[valid_mask]
        }

    def reconstruct_3d_points(self, kp_left, disparity):
        """使用 Q 矩阵将视差图重投影为三维点云"""
        if len(kp_left) == 0:
            return np.array([])

        # --- 最终修复 Part 1: 将坐标和视差从模型尺寸还原到ROI尺寸 ---
        orig_h, orig_w = self.roi_left[3], self.roi_left[2]
        scale_x = orig_w / self.cfg.IMAGE_WIDTH
        scale_y = orig_h / self.cfg.IMAGE_HEIGHT

        kp_left_roi = kp_left.copy()
        disparity_roi = disparity.copy()

        kp_left_roi[:, 0] *= scale_x
        kp_left_roi[:, 1] *= scale_y
        disparity_roi *= scale_x

        # --- 最终修复 Part 2: 将ROI坐标和视差转换到完整图像坐标系 ---
        # 2a. 调整 (x, y) 坐标
        kp_left_full = kp_left_roi.copy()
        kp_left_full[:, 0] += self.roi_left[0]
        kp_left_full[:, 1] += self.roi_left[1]

        # 2b. 调整视差, 这是之前遗漏的关键步骤
        # d_full = d_roi + (roi_offset_x_left - roi_offset_x_right)
        disparity_full = disparity_roi + (self.roi_left[0] - self.roi_right[0])

        # --- 最终修复 Part 3: 使用完整坐标和原始Q矩阵进行重建 ---
        points_2d = np.zeros((len(kp_left_full), 3), dtype=np.float32)
        points_2d[:, :2] = kp_left_full
        points_2d[:, 2] = disparity_full

        # 使用原始的、未经修改的Q矩阵
        points_3d = cv2.reprojectImageTo3D(points_2d, self.Q)

        # 增加 z > 0 的过滤条件，移除相机后方的无效点
        valid_mask = (points_3d[:, 2] < 50000) & (np.abs(points_3d[:, 0]) < 50000) & (
                    np.abs(points_3d[:, 1]) < 50000) & (points_3d[:, 2] > 0)
        return points_3d[valid_mask]

    def visualize_matches(self, img_left, img_right, kp_left, disparity, save_path):
        """可视化二维匹配结果"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        axes[0].imshow(img_left, cmap='gray')
        axes[0].scatter(kp_left[:, 0], kp_left[:, 1], c='r', s=15, alpha=0.7)
        axes[0].set_title(f"左图检测到的关键点 ({len(kp_left)}个)")
        axes[0].axis('off')
        h, w = img_left.shape
        combined_img = np.hstack((img_left, img_right))
        axes[1].imshow(combined_img, cmap='gray')
        axes[1].set_title("关键点匹配")
        axes[1].axis('off')
        for i in range(len(kp_left)):
            p1 = (int(kp_left[i, 0]), int(kp_left[i, 1]))
            p2 = (int(kp_left[i, 0] - disparity[i] + w), int(kp_left[i, 1]))
            color = np.random.rand(3, )
            axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=0.8)
            axes[1].scatter(p1[0], p1[1], c=[color], s=10)
            axes[1].scatter(p2[0], p2[1], c=[color], s=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"二维可视化图像已保存至: {save_path}")

    def visualize_3d_point_cloud(self, points_3d, save_path):
        """使用 Plotly 创建可交互的三维点云图"""
        if points_3d.shape[0] < 5:  # 为稳健计算百分位数，需要少量点
            print("没有足够的有效3D点可供可视化。")
            return

        # --- 使用百分位数过滤几何上的离群点 ---
        z_lower_bound = np.percentile(points_3d[:, 2], 1)
        z_upper_bound = np.percentile(points_3d[:, 2], 99)

        inliers_mask = (points_3d[:, 2] >= z_lower_bound) & (points_3d[:, 2] <= z_upper_bound)
        points_3d_inliers = points_3d[inliers_mask]

        if points_3d_inliers.shape[0] == 0:
            print("过滤离群点后没有剩余的点可供可视化。")
            return

        center = points_3d_inliers.mean(axis=0)
        points_centered = points_3d_inliers - center
        x, y, z = points_centered[:, 0], points_centered[:, 1], points_centered[:, 2]

        # --- 最终修复：手动设置颜色范围以忽略边缘效应 ---
        # 计算居中后Z值的百分位数，作为颜色映射的边界
        cmin = np.percentile(z, 2)
        cmax = np.percentile(z, 98)
        # --- 修复结束 ---

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=z,
                colorscale='Viridis',
                opacity=0.8,
                # 手动设置颜色条的上下限，强制可视化工具关注核心数据范围
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title='Z (相对高度)')
            )
        )])
        fig.update_layout(
            title='重建的三维点云 (已居中并过滤)',
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title='X 轴 (相对单位)',
                yaxis_title='Y 轴 (相对单位)',
                zaxis_title='Z 轴 (相对单位)',
                aspectmode='data',  # 使用 'data' 以更好地反映真实比例
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8)
                )
            )
        )
        fig.write_html(save_path)
        print(f"三维点云交互式文件已保存至: {save_path}")


def find_latest_run_dir():
    """自动查找并返回最新的训练运行目录"""
    runs_base_dir = r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_sparse"
    if not os.path.exists(runs_base_dir):
        print(f"错误: 基础运行目录不存在 '{runs_base_dir}'")
        return None
    all_runs = [d for d in os.listdir(runs_base_dir) if os.path.isdir(os.path.join(runs_base_dir, d))]
    if not all_runs:
        print(f"错误: 在 '{runs_base_dir}' 中没有找到任何运行目录。")
        return None
    latest_run_name = sorted(all_runs)[-1]
    latest_run_path = os.path.join(runs_base_dir, latest_run_name)
    print(f"自动检测到最新的运行目录: {latest_run_path}")
    return latest_run_path


def main():
    latest_run_dir = find_latest_run_dir()
    if not latest_run_dir:
        print("评估中止。请确保 'runs_base_dir' 路径正确，并且至少已成功运行过一次训练。")
        return

    IMAGE_INDICES_TO_EVALUATE = [100, 250, 400]
    evaluator = Evaluator(run_dir=latest_run_dir)
    first_html_path = None
    for idx in IMAGE_INDICES_TO_EVALUATE:
        print(f"\n--- 正在处理图像索引: {idx} ---")
        try:
            image_data = evaluator.load_and_preprocess_image(idx)
            inference_results = evaluator.run_inference(image_data)
            points_3d = evaluator.reconstruct_3d_points(
                inference_results['kp_left'],
                inference_results['disparity']
            )
            base_filename = f"eval_idx_{idx}"
            vis_2d_path = os.path.join(evaluator.output_dir, f"{base_filename}_matches.png")
            vis_3d_path = os.path.join(evaluator.output_dir, f"{base_filename}_point_cloud.html")
            evaluator.visualize_matches(
                image_data['display_left'],
                image_data['display_right'],
                inference_results['kp_left'],
                inference_results['disparity'],
                vis_2d_path
            )
            evaluator.visualize_3d_point_cloud(points_3d, vis_3d_path)
            if first_html_path is None and os.path.exists(vis_3d_path):
                first_html_path = vis_3d_path
        except Exception as e:
            print(f"处理图像索引 {idx} 时发生错误: {e}")

    if first_html_path:
        print(f"\n评估完成。正在尝试用默认浏览器打开: {first_html_path}")
        webbrowser.open(f'file://{os.path.realpath(first_html_path)}')
    else:
        print("\n评估完成，但没有生成可打开的三维可视化文件。")


if __name__ == "__main__":
    main()

