# create_animation.py
# 描述: 该脚本加载训练好的模型，处理一个连续的图像序列，
# 为每一帧重建三维点云，并通过平面拟合与调平技术校正视角，
# 最终为“侧方”和“俯瞰”两种视角分别生成动画视频。
#
# v2 更新:
# - 修复了 FileNotFoundError，脚本现在会自动创建 'evaluation_results' 目录。
# - 更新了 import 语句，以匹配最新的训练脚本 'sparse_reconstructor_1022_gemini.py'。
#
# v3 更新:
# - 更新 import 语句以优先匹配 'sparse_reconstructor_1023_gemini.py'。
#
# v4 更新:
# - 在 plot_frame 中添加 'disparity' 视图模式，用于可视化2D视差图。
# - 在 main 函数中添加生成视差视频的逻辑。

import os
import sys
import glob
import json
import dataclasses
from dataclasses import dataclass
import shutil

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # 用于颜色映射
from tqdm import tqdm
# 修正 DeprecationWarning: 明确使用 v2 版本的 imageio API
import imageio.v2 as imageio

# 新增导入：用于平面拟合与正交校正
from sklearn.decomposition import PCA

# --- 中文显示配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# --- 配置结束 ---


# --- v3 MODIFICATION: 更新 import 以优先匹配最新的训练脚本 ---
try:
    # 优先导入用户指定的最新脚本名
    from sparse_reconstructor_1023_gemini import Config, SparseMatchingStereoModel

    print("成功导入 'sparse_reconstructor_1023_gemini.py'")
except ImportError:
    try:
        # 否则，尝试导入之前的版本
        from sparse_reconstructor_1022_gemini import Config, SparseMatchingStereoModel

        print("警告: 未找到 '1023' 版本, 已导入 'sparse_reconstructor_1022_gemini.py'")
    except ImportError:
        try:
            # 再尝试导入原始文件名
            from sparse_reconstructor_1013_gemini import Config, SparseMatchingStereoModel

            print("警告: 未找到 '1023' 或 '1022' 版本, 已导入 'sparse_reconstructor_1013_gemini.py'")
        except ImportError:
            print("错误: 无法导入任何版本的训练脚本 ('1023', '1022', '1013').")
            print("请确保此脚本与您的训练脚本在同一个文件夹中。")
            sys.exit(1)


# --- END MODIFICATION ---


class Animator:
    """负责生成三维点云动画的类"""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.cfg = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载相机标定参数
        try:
            calib = np.load(self.cfg.CALIBRATION_FILE)
            self.Q = calib['Q']
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            self.roi_left = tuple(calib['roi_left'])  # (x, y, w, h)
            self.roi_right = tuple(calib['roi_right'])  # (x, y, w, h)

        except Exception as e:
            sys.exit(f"加载相机标定文件失败: {e}")

        # 初始化模型并加载权重
        model_path = os.path.join(self.run_dir, "checkpoints", "best_model_sparse.pth")
        if not os.path.exists(model_path):
            sys.exit(f"错误: 找不到模型文件 '{model_path}'。请确保模型已训练。")

        self.model = SparseMatchingStereoModel(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"模型已从 '{model_path}' 加载。")

        # 获取图像文件列表
        self.left_images = sorted(glob.glob(os.path.join(self.cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"在 '{self.cfg.LEFT_IMAGE_DIR}' 中未找到图像。")

        # --- v4 添加: 预计算用于视差可视化的全局范围 ---
        self.global_disparity_min = float('inf')
        self.global_disparity_max = float('-inf')
        # --- END v4 添加 ---

    def _load_config(self) -> Config:
        log_file = os.path.join(self.run_dir, "logs", "training_log.json")
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            config_dict = log_data['config']
            config_fields = {f.name for f in dataclasses.fields(Config)}
            # --- Robustness: Filter config_dict based on Config fields ---
            filtered_config_dict = {k: v for k, v in config_dict.items() if k in config_fields}
            # Handle potential missing fields if loading config from older runs
            default_config = Config()
            for field in dataclasses.fields(Config):
                if field.name not in filtered_config_dict:
                    print(
                        f"警告: 在日志文件中未找到配置 '{field.name}'，使用默认值: {getattr(default_config, field.name)}")
                    filtered_config_dict[field.name] = getattr(default_config, field.name)

            # --- v4 添加: 确保 IMAGE_HEIGHT/WIDTH 与加载的配置匹配 ---
            # 如果 Animator 初始化后 cfg 改变，确保这里使用的是正确的尺寸
            self.image_height_from_cfg = filtered_config_dict.get('IMAGE_HEIGHT', default_config.IMAGE_HEIGHT)
            self.image_width_from_cfg = filtered_config_dict.get('IMAGE_WIDTH', default_config.IMAGE_WIDTH)
            # --- END v4 添加 ---

            return Config(**filtered_config_dict)
            # --- End Robustness ---
        except FileNotFoundError:
            sys.exit(f"错误: 找不到日志文件 '{log_file}'。无法加载配置。")
        except Exception as e:
            sys.exit(f"加载配置文件 '{log_file}' 失败: {e}")

    # --- v4 添加: 新增函数用于预计算视差范围 ---
    def precompute_disparity_range(self, frame_indices):
        """遍历指定帧以确定全局视差范围，用于一致的可视化。"""
        print("\n--- 步骤 0/6: 正在预计算全局视差范围 ---")
        min_disp = float('inf')
        max_disp = float('-inf')
        for i in tqdm(frame_indices, desc="预计算视差"):
            _, _, _, disparity_roi = self._get_model_output(i)  # 只关心视差
            if disparity_roi is not None and len(disparity_roi) > 0:
                min_disp = min(min_disp, np.min(disparity_roi))
                max_disp = max(max_disp, np.max(disparity_roi))

        if np.isinf(min_disp) or np.isinf(max_disp):
            print("警告: 未能计算有效的视差范围。将使用默认值。")
            self.global_disparity_min = 0
            self.global_disparity_max = 50  # 假设一个合理的范围
        else:
            self.global_disparity_min = min_disp
            self.global_disparity_max = max_disp
            # 添加一些padding
            padding = (self.global_disparity_max - self.global_disparity_min) * 0.05
            self.global_disparity_min -= padding
            self.global_disparity_max += padding
            print(f"全局视差范围已计算: [{self.global_disparity_min:.2f}, {self.global_disparity_max:.2f}]")

    # --- END v4 添加 ---

    def _get_model_output(self, image_index: int):
        """将模型推理与坐标转换逻辑提取到一个单独的函数中。"""
        left_path = self.left_images[image_index]
        right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, 'right' + os.path.basename(left_path)[4:])

        left_raw = cv2.imread(left_path, 0)
        right_raw = cv2.imread(right_path, 0)

        if left_raw is None or right_raw is None: return None, None, None, None

        left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

        lx, ly, lw, lh = self.roi_left
        left_rect_cropped = left_rect[ly:ly + lh, lx:lx + lw]
        rx, ry, rw, rh = self.roi_right
        # right_rect_cropped = right_rect[ry:ry + rh, rx:rx + rw] # right crop 不用于推理

        # --- v4 修改: 使用加载的配置中的尺寸 ---
        left_img_resized = cv2.resize(left_rect_cropped, (self.image_width_from_cfg, self.image_height_from_cfg))
        right_img_resized = cv2.resize(right_rect[ry:ry + rh, rx:rx + rw],
                                       (self.image_width_from_cfg, self.image_height_from_cfg))
        # --- END v4 修改 ---

        with torch.no_grad():
            left_gray_tensor = torch.from_numpy(left_img_resized).float().unsqueeze(0).unsqueeze(0).to(
                self.device) / 255.0
            right_gray_tensor = torch.from_numpy(right_img_resized).float().unsqueeze(0).unsqueeze(0).to(
                self.device) / 255.0
            left_rgb_tensor = torch.from_numpy(
                cv2.cvtColor(left_img_resized, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float().unsqueeze(0).to(
                self.device) / 255.0
            right_rgb_tensor = torch.from_numpy(
                cv2.cvtColor(right_img_resized, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float().unsqueeze(0).to(
                self.device) / 255.0
            mask = torch.ones_like(left_gray_tensor)
            outputs = self.model(left_gray_tensor, right_gray_tensor, left_rgb_tensor, right_rgb_tensor, mask)

        kp_left = outputs['keypoints_left'][0].cpu().numpy()
        scores_left = outputs['scores_left'][0].cpu().numpy()
        disparity = outputs['disparity'][0].cpu().numpy()

        valid_mask = scores_left > 0.1
        kp_left_valid = kp_left[valid_mask]
        disparity_valid = disparity[valid_mask]

        if len(kp_left_valid) == 0: return None, None, None, None

        orig_h, orig_w = self.roi_left[3], self.roi_left[2]
        # --- v4 修改: 使用加载的配置中的尺寸 ---
        scale_x = orig_w / self.image_width_from_cfg
        scale_y = orig_h / self.image_height_from_cfg
        # --- END v4 修改 ---

        kp_left_roi = kp_left_valid.copy()
        disparity_roi = disparity_valid.copy()

        kp_left_roi[:, 0] *= scale_x
        kp_left_roi[:, 1] *= scale_y
        disparity_roi *= scale_x  # 视差只受 X 尺度影响

        kp_left_full = kp_left_roi.copy()
        kp_left_full[:, 0] += self.roi_left[0]
        kp_left_full[:, 1] += self.roi_left[1]
        disparity_full = disparity_roi + (self.roi_left[0] - self.roi_right[0])

        # 返回需要的值，包括原始左图裁剪区域用于可视化
        return left_rect_cropped, kp_left_full, disparity_full, disparity_roi  # 添加 disparity_roi

    def process_frame_for_3d(self, image_index: int):
        """仅执行3D重建并返回点云。"""
        _, kp_left_full, disparity_full, _ = self._get_model_output(image_index)

        if kp_left_full is None: return None

        # 3D 重建逻辑 (来自旧 process_frame)
        cx = -self.Q[0, 3]
        cy = -self.Q[1, 3]
        f = self.Q[2, 3]
        Tx = -1 / self.Q[3, 2]
        Tx = np.abs(Tx)

        x = kp_left_full[:, 0]
        y = kp_left_full[:, 1]
        d = disparity_full

        # 处理零视差和非常小的视差以避免除零
        valid_disp_mask = d > 1e-3  # 阈值可以调整
        x = x[valid_disp_mask]
        y = y[valid_disp_mask]
        d = d[valid_disp_mask]

        if len(d) == 0: return None

        Z = f * Tx / d
        X = (x - cx) * Z / f
        Y = (y - cy) * Z / f

        points_3d = np.stack([X, Y, Z], axis=-1)

        # 进一步过滤掉极端值
        valid_mask_3d = (Z < 50000) & (np.abs(X) < 50000) & (np.abs(Y) < 50000) & (Z > 0)
        final_points = points_3d[valid_mask_3d]

        return final_points if len(final_points) > 0 else None

    # --- v4 修改: 添加了 disparity 模式 ---
    def plot_frame(self, data_to_plot, save_path, bounds, frame_idx, view_mode='side'):
        """根据视角模式，绘制 2D 侧方剖面、俯瞰高度图或 2D 视差图"""
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)  # 主轴

        # 处理空数据情况
        if data_to_plot is None or \
                (isinstance(data_to_plot, tuple) and (
                        data_to_plot[0] is None or data_to_plot[1] is None or data_to_plot[2] is None)) or \
                (isinstance(data_to_plot, np.ndarray) and data_to_plot.shape[0] == 0):

            title_prefix = "三维重建动态"
            if view_mode == 'top':
                title = f'{title_prefix} (正交俯瞰高度图): 帧 {frame_idx} (无数据)'
            elif view_mode == 'side':
                title = f'{title_prefix} (侧方剖面图): 帧 {frame_idx} (无数据)'
            elif view_mode == 'disparity':
                title = f'2D 视差图: 帧 {frame_idx} (无有效点)'
            else:
                title = f'帧 {frame_idx} (无数据)'

            ax.set_title(title)
            if view_mode == 'top': ax.set_aspect('equal', adjustable='box')
            plt.savefig(save_path, dpi=96);
            plt.close(fig)
            return

        # --- 新增: 视差可视化模式 ---
        if view_mode == 'disparity':
            left_img_roi, kp_roi, disp_roi = data_to_plot  # 解包数据
            ax.imshow(left_img_roi, cmap='gray')
            # 使用全局范围进行着色
            sc = ax.scatter(kp_roi[:, 0], kp_roi[:, 1], c=disp_roi, cmap='viridis', s=20, alpha=0.7,
                            vmin=self.global_disparity_min, vmax=self.global_disparity_max)
            plt.colorbar(sc, label='视差 (像素)')
            ax.set_title(f'2D 视差图: 帧 {frame_idx}')
            ax.axis('off')  # 通常不需要坐标轴

        # --- 现有: 3D 可视化模式 ---
        elif view_mode in ['top', 'side']:
            points_to_plot = data_to_plot  # 重命名以便清晰
            center = bounds['center']
            p_centered = points_to_plot - center

            if view_mode == 'top':
                ax.set_title(f'三维重建动态 (正交俯瞰高度图): 帧 {frame_idx}')
                sc = ax.scatter(p_centered[:, 0], p_centered[:, 1], c=p_centered[:, 2], cmap='viridis', s=15,
                                vmin=bounds['zmin_color'], vmax=bounds['zmax_color'])
                plt.colorbar(sc, label='Z (相对高度)')
                ax.set_xlim(bounds['xmin_plot'], bounds['xmax_plot'])
                ax.set_ylim(bounds['ymin_plot'], bounds['ymax_plot'])
                ax.set_xlabel('X (主方向)')
                ax.set_ylabel('Y (次方向)')
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True)
            else:  # 'side'
                ax.set_title(f'三维重建动态 (侧方平均剖面图): 帧 {frame_idx}')
                if p_centered.shape[0] > 10:
                    num_bins = 100
                    x_min, x_max = p_centered[:, 0].min(), p_centered[:, 0].max()
                    # 确保 x_min 和 x_max 不是 NaN 或 Inf
                    if not (np.isfinite(x_min) and np.isfinite(x_max)):
                        print(f"警告: 帧 {frame_idx} 的 X 范围无效，跳过侧视图平均。")
                        sc = ax.scatter(p_centered[:, 0], p_centered[:, 2], c=p_centered[:, 2], cmap='viridis', s=15,
                                        vmin=bounds['zmin_color'], vmax=bounds['zmax_color'])
                    else:
                        bins = np.linspace(x_min, x_max, num_bins)
                        binned_z = np.zeros(num_bins - 1)
                        bin_centers_x = (bins[:-1] + bins[1:]) / 2
                        digitized = np.digitize(p_centered[:, 0], bins)

                        for i in range(1, len(bins)):
                            points_in_bin = p_centered[digitized == i]
                            if len(points_in_bin) > 0:
                                binned_z[i - 1] = points_in_bin[:, 2].mean()
                            else:
                                binned_z[i - 1] = np.nan
                        valid_bins = ~np.isnan(binned_z)

                        # --- v4 改进: 如果没有有效 bin，则回退到散点图 ---
                        if np.any(valid_bins):
                            sc = ax.scatter(bin_centers_x[valid_bins], binned_z[valid_bins], c=binned_z[valid_bins],
                                            cmap='viridis',
                                            s=20, vmin=bounds['zmin_color'], vmax=bounds['zmax_color'])
                        else:
                            print(f"警告: 帧 {frame_idx} 在侧视图分箱后没有有效数据点，回退到散点图。")
                            sc = ax.scatter(p_centered[:, 0], p_centered[:, 2], c=p_centered[:, 2], cmap='viridis',
                                            s=15,
                                            vmin=bounds['zmin_color'], vmax=bounds['zmax_color'])
                        # --- END v4 改进 ---

                else:
                    sc = ax.scatter(p_centered[:, 0], p_centered[:, 2], c=p_centered[:, 2], cmap='viridis', s=15,
                                    vmin=bounds['zmin_color'], vmax=bounds['zmax_color'])

                ax.set_xlim(bounds['xmin_plot'], bounds['xmax_plot'])
                ax.set_ylim(bounds['zmin_amplified'], bounds['zmax_amplified'])
                ax.set_xlabel('X (主方向 / 水槽长度)')
                ax.set_ylabel('Z (相对高度 / 波高)')
                ax.grid(True)
                ax.set_aspect('auto')
                plt.colorbar(sc, label='Z (相对高度)')
        else:
            print(f"错误: 未知的 view_mode '{view_mode}'")
            ax.set_title(f'错误: 未知视图模式 - 帧 {frame_idx}')

        plt.savefig(save_path, dpi=96)
        plt.close(fig)
    # --- END v4 修改 ---


def find_latest_run_dir():
    # 假设你的项目根目录是 D:\Research\wave_reconstruction_project\DINOv3
    PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
    runs_base_dir = os.path.join(PROJECT_ROOT, "training_runs_sparse")

    if not os.path.exists(runs_base_dir):
        print(f"错误: 找不到训练目录 '{runs_base_dir}'")
        return None

    all_runs = [d for d in os.listdir(runs_base_dir) if os.path.isdir(os.path.join(runs_base_dir, d))]
    if not all_runs:
        print(f"错误: 在 '{runs_base_dir}' 中没有找到任何训练运行。")
        return None

    latest_run_name = sorted(all_runs)[-1]
    latest_run_path = os.path.join(runs_base_dir, latest_run_name)
    print(f"自动检测到最新的运行目录: {latest_run_path}")
    return latest_run_path


def get_leveling_rotation(points):
    """使用主成分分析(PCA)计算旋转矩阵以使点云平面水平"""
    if points is None or points.shape[0] < 10:
        print("警告: 点云过少 (<10)，无法进行调平，返回单位矩阵。")
        return np.identity(3)

    try:
        pca = PCA(n_components=3)
        pca.fit(points)
        normal = pca.components_[2]

        if normal[2] < 0:
            normal = -normal

        target_normal = np.array([0, 0, 1.0])

        if np.allclose(normal, target_normal):
            return np.identity(3)

        axis = np.cross(normal, target_normal)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            if np.allclose(normal, -target_normal):
                # Handle 180 degree rotation case
                return np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])  # Or choose another axis, e.g., around X
            else:
                # Already aligned
                return np.identity(3)

        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(normal, target_normal), -1.0, 1.0))

        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        rotation_matrix = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # Sanity check for NaN/Inf in rotation matrix
        if not np.all(np.isfinite(rotation_matrix)):
            print("警告: 计算出的调平旋转矩阵包含无效值，返回单位矩阵。")
            return np.identity(3)

        return rotation_matrix
    except Exception as e:
        print(f"警告: 计算调平旋转时出错: {e}，返回单位矩阵。")
        return np.identity(3)


def get_ortho_rotation(points):
    """计算旋转矩阵以使点云主方向与X轴对齐"""
    if points is None or points.shape[0] < 3:
        print("警告: 点云过少 (<3)，无法进行正交旋转，返回单位矩阵。")
        return np.identity(3)

    try:
        # Only fit PCA on X and Y coordinates
        pca = PCA(n_components=2)
        pca.fit(points[:, :2])
        main_axis = pca.components_[0]
        angle = np.arctan2(main_axis[1], main_axis[0])

        # Create 3D rotation matrix around Z-axis
        cos_angle = np.cos(-angle)
        sin_angle = np.sin(-angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])

        # Sanity check for NaN/Inf
        if not np.all(np.isfinite(rotation_matrix)):
            print("警告: 计算出的正交旋转矩阵包含无效值，返回单位矩阵。")
            return np.identity(3)

        return rotation_matrix
    except Exception as e:
        print(f"警告: 计算正交旋转时出错: {e}，返回单位矩阵。")
        return np.identity(3)


def main():
    latest_run_dir = find_latest_run_dir()
    if not latest_run_dir:
        print("未找到最新的运行目录，脚本终止。")
        return

    FRAME_START, FRAME_END, FPS = 100, 300, 3
    frame_indices = range(FRAME_START, FRAME_END)  # 用于预计算

    animator = Animator(run_dir=latest_run_dir)

    # --- v4 添加: 预计算视差范围 ---
    animator.precompute_disparity_range(frame_indices)
    # --- END v4 添加 ---

    # --- v4 修改: 添加视差帧目录 ---
    frames_side_dir = os.path.join(latest_run_dir, "frames_side")
    frames_top_dir = os.path.join(latest_run_dir, "frames_top")
    frames_disp_dir = os.path.join(latest_run_dir, "frames_disparity")  # 新目录
    for d in [frames_side_dir, frames_top_dir, frames_disp_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)
    # --- END v4 修改 ---

    print("\n--- 步骤 1/6: 正在处理所有帧以收集点云数据 ---")
    all_points = []
    # --- v4 修改: 分离模型推理和3D重建 ---
    frame_data_for_disp = []  # 存储视差可视化所需数据
    for i in tqdm(frame_indices, desc="处理帧 (模型推理)"):
        # 先获取模型输出和用于可视化的数据
        left_roi, kp_full, disp_full, disp_roi = animator._get_model_output(i)
        if kp_full is not None:
            # 存储视差可视化所需数据 (使用 ROI 坐标)
            kp_roi_coords = kp_full.copy()
            kp_roi_coords[:, 0] -= animator.roi_left[0]
            kp_roi_coords[:, 1] -= animator.roi_left[1]
            frame_data_for_disp.append((left_roi, kp_roi_coords, disp_roi))  # 存储 ROI 内坐标和视差

            # 然后进行3D重建
            points_3d = animator.process_frame_for_3d(i)  # 使用独立函数
            if points_3d is not None:
                all_points.append(points_3d)
        else:
            # 如果模型无输出，也添加占位符，保持帧数一致
            frame_data_for_disp.append((None, None, None))
            all_points.append(None)  # 添加 None 占位

    # 过滤掉 None 值，用于后续 PCA 计算
    valid_point_clouds = [p for p in all_points if p is not None]
    if not valid_point_clouds:
        print("错误: 在指定范围内没有重建出任何有效的三维点。");
        return
    full_point_cloud = np.vstack(valid_point_clouds)
    # --- END v4 修改 ---

    print("\n--- 步骤 2/6: 正在计算最终旋转矩阵（调平+正交） ---")
    leveling_rotation = get_leveling_rotation(full_point_cloud)
    # 应用调平旋转后再计算正交旋转
    cloud_leveled = full_point_cloud @ leveling_rotation.T
    ortho_rotation = get_ortho_rotation(cloud_leveled)
    final_rotation = ortho_rotation @ leveling_rotation  # 最终旋转 = 先调平，再正交

    print("\n--- 步骤 3/6: 正在应用变换、通过百分位数过滤离群点 (全局) ---")
    final_cloud = full_point_cloud @ final_rotation.T  # 应用最终旋转到原始点云

    # 仅使用有限值计算百分位数，避免 NaN/Inf 污染
    finite_z = final_cloud[:, 2][np.isfinite(final_cloud[:, 2])]
    if len(finite_z) < 2:
        print("警告: 有效 Z 值过少，无法计算百分位数，使用默认 Z 范围。")
        z_lower_bound = np.mean(finite_z) - 100 if len(finite_z) > 0 else -100
        z_upper_bound = np.mean(finite_z) + 100 if len(finite_z) > 0 else 100
    else:
        z_lower_bound = np.percentile(finite_z, 1)
        z_upper_bound = np.percentile(finite_z, 99)

    # 对整个点云应用 Z 范围过滤，得到用于计算边界的内点
    inliers_mask = (final_cloud[:, 2] >= z_lower_bound) & (final_cloud[:, 2] <= z_upper_bound) & np.isfinite(
        final_cloud[:, 2])
    final_cloud_inliers = final_cloud[inliers_mask]

    # 如果过滤后点太少，发出警告
    if final_cloud_inliers.shape[0] < 10:
        print("警告: 全局 Z 百分位数过滤后剩余点过少 (<10)，边界计算可能不准确。")
        # 可以选择回退到使用未过滤的点计算边界，或者接受可能不准的边界
        # 这里我们还是用过滤后的点，但用户需知晓风险
        if final_cloud_inliers.shape[0] == 0:
            print("错误: 过滤后无剩余点，无法计算边界，脚本终止。")
            return

    print("\n--- 步骤 4/6: 正在计算最终的可视化边界与比例 (基于全局内点) ---")
    # 确保使用有限值计算 min/max
    finite_x_inliers = final_cloud_inliers[:, 0][np.isfinite(final_cloud_inliers[:, 0])]
    finite_y_inliers = final_cloud_inliers[:, 1][np.isfinite(final_cloud_inliers[:, 1])]
    finite_z_inliers = final_cloud_inliers[:, 2][np.isfinite(final_cloud_inliers[:, 2])]

    # 如果某个维度没有有限值，则无法计算边界
    if not (len(finite_x_inliers) > 0 and len(finite_y_inliers) > 0 and len(finite_z_inliers) > 0):
        print("错误: 过滤后的内点中存在维度没有有效数值，无法计算边界，脚本终止。")
        return

    bounds = {
        'xmin': finite_x_inliers.min(), 'xmax': finite_x_inliers.max(),
        'ymin': finite_y_inliers.min(), 'ymax': finite_y_inliers.max(),
        'zmin': finite_z_inliers.min(), 'zmax': finite_z_inliers.max(),
    }
    # 使用过滤后的内点计算中心
    bounds['center'] = np.mean(final_cloud_inliers, axis=0)
    # 确保中心点也是有限的
    if not np.all(np.isfinite(bounds['center'])):
        print("警告: 计算出的中心点包含无效值，将使用 [0, 0, 0] 代替。")
        bounds['center'] = np.array([0.0, 0.0, 0.0])

    # 定义绘图的padding (基于范围，如果范围为0则给一个固定padding)
    range_x = bounds['xmax'] - bounds['xmin']
    range_y = bounds['ymax'] - bounds['ymin']
    range_z = bounds['zmax'] - bounds['zmin']
    padding_x = range_x * 0.1 if range_x > 1e-6 else 10.0
    padding_y = range_y * 0.1 if range_y > 1e-6 else 10.0
    # padding_z = range_z * 0.1 if range_z > 1e-6 else 10.0 # Z padding 在下面计算

    # 设置常规的坐标轴极限
    bounds.update({
        'xmin_plot': bounds['xmin'] - padding_x - bounds['center'][0],
        'xmax_plot': bounds['xmax'] + padding_x - bounds['center'][0],
        'ymin_plot': bounds['ymin'] - padding_y - bounds['center'][1],
        'ymax_plot': bounds['ymax'] + padding_y - bounds['center'][1],
    })

    # 设置颜色条的范围, 保持数据真实性
    bounds.update({
        'zmin_color': bounds['zmin'] - bounds['center'][2],
        'zmax_color': bounds['zmax'] - bounds['center'][2]
    })

    # 为侧方剖面图单独计算放大的Z轴范围
    z_amplification_factor = 8.0  # 可以调整这个因子
    # 确保 zmin_color 和 zmax_color 是有限的
    if not (np.isfinite(bounds['zmin_color']) and np.isfinite(bounds['zmax_color'])):
        print("警告: Z 颜色范围无效，侧视图放大可能不准确。")
        z_mean_centered = 0
        z_range_half_amplified = 100  # 给一个默认范围
    else:
        color_range_z = bounds['zmax_color'] - bounds['zmin_color']
        if color_range_z < 1e-6:  # 如果范围几乎为0
            print("警告: Z 颜色范围过小，侧视图放大可能不准确。")
            z_mean_centered = bounds['zmin_color']
            z_range_half_amplified = 10  # 给一个默认半范围
        else:
            z_mean_centered = bounds['zmin_color'] + color_range_z / 2
            z_range_half_amplified = color_range_z / 2 * z_amplification_factor

    bounds['zmin_amplified'] = z_mean_centered - z_range_half_amplified
    bounds['zmax_amplified'] = z_mean_centered + z_range_half_amplified

    print("\n--- 步骤 5/6: 正在逐帧生成并合成视频 ---")
    frame_files_side, frame_files_top, frame_files_disp = [], [], []  # 添加视差文件列表

    # --- v4 修改: 使用 zip 迭代原始点云和视差数据 ---
    # all_points 包含了 None 占位符
    # frame_data_for_disp 也包含了 (None, None, None) 占位符
    if len(all_points) != len(frame_data_for_disp):
        print(f"错误: 点云列表 ({len(all_points)}) 和视差数据列表 ({len(frame_data_for_disp)}) 长度不匹配!")
        return

    frame_iterator = tqdm(zip(all_points, frame_data_for_disp), total=len(all_points), desc="生成帧")

    for i, (points, disp_data) in enumerate(frame_iterator):
        frame_idx = FRAME_START + i

        # --- 生成 3D 视图帧 ---
        points_to_plot_3d = None
        if points is not None:
            points_rotated = points @ final_rotation.T
            # 对 *当前帧* 应用 Z 范围过滤
            frame_inliers_mask = (points_rotated[:, 2] >= z_lower_bound) & (
                        points_rotated[:, 2] <= z_upper_bound) & np.isfinite(points_rotated[:, 2])
            if np.any(frame_inliers_mask):
                points_to_plot_3d = points_rotated[frame_inliers_mask]

        # 保存侧视图帧
        save_path_side = os.path.join(frames_side_dir, f"frame_{i:04d}.png")
        animator.plot_frame(points_to_plot_3d, save_path_side, bounds, frame_idx, view_mode='side')
        frame_files_side.append(save_path_side)

        # 保存俯视图帧
        save_path_top = os.path.join(frames_top_dir, f"frame_{i:04d}.png")
        animator.plot_frame(points_to_plot_3d, save_path_top, bounds, frame_idx, view_mode='top')
        frame_files_top.append(save_path_top)

        # --- 生成 2D 视差视图帧 ---
        # disp_data 是 (left_roi, kp_roi, disp_roi) 或 (None, None, None)
        save_path_disp = os.path.join(frames_disp_dir, f"frame_{i:04d}.png")
        # 确保 kp_roi 和 disp_roi 都是有效的 numpy 数组
        if disp_data[0] is not None and isinstance(disp_data[1], np.ndarray) and isinstance(disp_data[2], np.ndarray):
            animator.plot_frame(disp_data, save_path_disp, None, frame_idx, view_mode='disparity')  # bounds 不需要
        else:
            # 如果没有有效数据，也创建一个空的图，保持帧数一致
            animator.plot_frame(None, save_path_disp, None, frame_idx, view_mode='disparity')
        frame_files_disp.append(save_path_disp)
    # --- END v4 修改 ---

    # --- v2 FIX: 确保输出目录存在 ---
    output_video_dir = os.path.join(latest_run_dir, "evaluation_results")
    os.makedirs(output_video_dir, exist_ok=True)  # exist_ok=True 使得如果目录已存在也不会报错
    print(f"\n输出目录已确认: {output_video_dir}")
    # --- END FIX ---

    # --- 合成侧视视频 ---
    video_path_side = os.path.join(output_video_dir, f"animation_side_view_fps{FPS}.mp4")
    try:
        with imageio.get_writer(video_path_side, fps=FPS, quality=8) as writer:  # quality 参数 (0-10)
            for filename in tqdm(frame_files_side, desc="合成侧视视频"):
                writer.append_data(imageio.imread(filename))
        print(f"侧视动画视频已成功生成！文件位置: {video_path_side}")
    except Exception as e:
        print(f"\n错误: 合成侧视视频失败: {e}")

    # --- 合成俯视视频 ---
    video_path_top = os.path.join(output_video_dir, f"animation_top_view_fps{FPS}.mp4")
    try:
        with imageio.get_writer(video_path_top, fps=FPS, quality=8) as writer:
            for filename in tqdm(frame_files_top, desc="合成俯视视频"):
                writer.append_data(imageio.imread(filename))
        print(f"俯视动画视频已成功生成！文件位置: {video_path_top}")
    except Exception as e:
        print(f"\n错误: 合成俯视视频失败: {e}")

    # --- v4 添加: 合成视差视频 ---
    video_path_disp = os.path.join(output_video_dir, f"animation_disparity_view_fps{FPS}.mp4")
    try:
        with imageio.get_writer(video_path_disp, fps=FPS, quality=8) as writer:
            for filename in tqdm(frame_files_disp, desc="合成视差视频"):
                writer.append_data(imageio.imread(filename))
        print(f"2D视差动画视频已成功生成！文件位置: {video_path_disp}")
    except Exception as e:
        print(f"\n错误: 合成视差视频失败: {e}")
    # --- END v4 添加 ---

    # 清理临时帧文件
    try:
        shutil.rmtree(frames_side_dir)
        shutil.rmtree(frames_top_dir)
        shutil.rmtree(frames_disp_dir)  # 清理视差帧
        print("\n临时帧文件已清理。")
    except Exception as e:
        print(f"\n警告: 清理临时帧文件失败: {e}")

    print("\n所有视频已生成。")


if __name__ == "__main__":
    main()

