# dinov3_path_b.py
# 实现了从含泡沫图像直接重建无泡沫波浪表面的端到端深度学习方案
# (最终稳定版：使用DINOv2模型，并修复了匹配算法的崩溃问题)

import os
import glob
import pickle
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import curve_fit, linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 使用稳定且技术同源的DINOv2模型 ---
try:
    from transformers import Dinov2Model
except ImportError:
    print("=" * 80)
    print("错误: 无法从 'transformers' 中导入 'Dinov2Model'。")
    print("这表明您的Conda环境依然存在问题。")
    print("\n请严格按照以下【环境修复指令】操作：")
    print("\n--- 环境修复指令 ---")
    print("1. 打开Anaconda Prompt终端。")
    print("2. 激活您的环境: \n   conda activate dino")
    print(
        "3. 安装或更新核心库: \n   pip install --upgrade torch torchvision transformers opencv-python scipy tqdm matplotlib scikit-learn")
    print("=" * 80)
    exit()


# --- 1. 配置中心 ---
@dataclass
class Config:
    """项目配置参数"""
    # --- 文件路径 (已根据您的环境配置) ---
    LEFT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/lresult/"
    RIGHT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/rresult/"
    CALIBRATION_FILE: str = "D:/Research/wave_reconstruction_project/camera_calibration/params/stereo_calib_params_from_matlab_full.npz"

    # --- 自动生成的数据路径 ---
    GROUND_TRUTH_DIR: str = "D:/Research/wave_reconstruction_project/data/ground_truth_disparity/"
    CHECKPOINT_DIR: str = "./checkpoints/"
    VISUALIZATION_DIR: str = "D:/Research/wave_reconstruction_project/data/visualization/"

    # --- 可视化控制开关 ---
    VISUALIZE_GT_GENERATION: bool = True
    VISUALIZE_FRAME_INTERVAL: int = 50

    # --- 数据处理参数 ---
    IMAGE_HEIGHT: int = 320
    IMAGE_WIDTH: int = 512

    # --- 模型与训练参数 ---
    DINO_MODEL_NAME: str = "facebook/dinov2-base"  # 使用DINOv2基础模型
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1


# --- 2. 真值生成器 (已添加可视化功能) ---
class GroundTruthGenerator:
    """
    自动化生成稠密、干净的视差图真值，并提供详细的可视化过程。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.GROUND_TRUTH_DIR, exist_ok=True)
        if cfg.VISUALIZE_GT_GENERATION:
            os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)
            print(f"可视化结果将保存在: {cfg.VISUALIZATION_DIR}")

        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            self.P1 = calib['P1']
            self.P2 = calib['P2']
            self.focal_length = self.P1[0, 0]
            self.baseline = -self.P2[0, 3] / self.P1[0, 0]
            print("相机标定参数加载成功。")
        except Exception as e:
            print(f"错误: 无法加载或解析标定文件 '{cfg.CALIBRATION_FILE}': {e}")
            raise

    @staticmethod
    def detect_markers(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.array([])
        _, binary_img = cv2.threshold(img, 68, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker_centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 8 < area < 500 and len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    hull = cv2.convexHull(contour)
                    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                    (w, h) = ellipse[1]
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100
                    if solidity > 0.85 and aspect_ratio < 8.0:
                        marker_centers.append(ellipse[0])
                except cv2.error:
                    continue
        return np.array(marker_centers, dtype=np.float32)

    def match_and_triangulate(self, left_markers: np.ndarray, right_markers: np.ndarray) -> tuple:
        if left_markers.shape[0] == 0 or right_markers.shape[0] == 0:
            return np.array([]), [], []

        cost_matrix = np.full((len(left_markers), len(right_markers)), np.inf)
        y_tolerance = 20

        for i, p_l in enumerate(left_markers):
            for j, p_r in enumerate(right_markers):
                if abs(p_l[1] - p_r[1]) < y_tolerance and p_l[0] > p_r[0]:
                    cost_matrix[i, j] = np.linalg.norm(p_l - p_r)

        # --- 关键修复：在调用匈牙利算法前检查成本矩阵是否可行 ---
        if not np.any(np.isfinite(cost_matrix)):
            print("  警告: 在当前帧中未找到满足几何约束的匹配对。")
            return np.array([]), [], []

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_points_l, matched_points_r = [], []
        for r, c in zip(row_ind, col_ind):
            if np.isfinite(cost_matrix[r, c]):
                matched_points_l.append(left_markers[r])
                matched_points_r.append(right_markers[c])

        if not matched_points_l:
            return np.array([]), [], []

        points_l_np = np.array(matched_points_l).T
        points_r_np = np.array(matched_points_r).T

        points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, points_l_np, points_r_np)
        points_3d = (points_4d_hom[:3] / (points_4d_hom[3] + 1e-8)).T
        return points_3d, matched_points_l, matched_points_r

    @staticmethod
    def wave_surface_func(xy_data, amplitude, freq_x, phase_x, freq_y, phase_y, offset):
        x, y = xy_data[0, :], xy_data[1, :]
        z = amplitude * np.cos(freq_x * x + phase_x) * np.cos(freq_y * y + phase_y) + offset
        return z

    def fit_and_generate_dense_map(self, points_3d: np.ndarray, orig_shape: tuple) -> tuple:
        if len(points_3d) < 10: return None, None
        x_data, y_data, z_data = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        try:
            initial_guess = [np.std(z_data), 0.1, 0, 0.1, 0, np.mean(z_data)]
            params, _ = curve_fit(self.wave_surface_func, np.vstack((x_data, y_data)), z_data, p0=initial_guess,
                                  maxfev=5000)
        except RuntimeError:
            return None, None
        h, w = orig_shape
        y_cam, x_cam = np.mgrid[0:h, 0:w]
        K_inv = np.linalg.inv(self.P1[:, :3])
        pixel_coords = np.stack([x_cam.flatten(), y_cam.flatten(), np.ones_like(x_cam.flatten())], axis=1)
        cam_coords = (K_inv @ pixel_coords.T).T
        world_x = cam_coords[:, 0] * np.mean(z_data)
        world_y = cam_coords[:, 1] * np.mean(z_data)
        dense_z = self.wave_surface_func(np.vstack((world_x, world_y)), *params)
        dense_z = dense_z.reshape((h, w))
        dense_z[dense_z <= 0] = 1e6
        disparity_map = (self.focal_length * self.baseline) / dense_z
        return disparity_map.astype(np.float32), params

    def process_all_frames(self):
        print("--- 开始生成或检查真值数据 ---")
        left_images = sorted(glob.glob(os.path.join(self.cfg.LEFT_IMAGE_DIR, "*.*")))
        for i in range(len(left_images)):
            frame_basename = os.path.basename(left_images[i])
            gt_path = os.path.join(self.cfg.GROUND_TRUTH_DIR, f"gt_disparity_{frame_basename.split('.')[0]}.pkl")
            print(f"\n--- [帧 {i + 1}/{len(left_images)}] 文件: {frame_basename} ---")
            if os.path.exists(gt_path):
                print("真值文件已存在，跳过生成。")
                continue

            print("步骤 1/4: 检测左右相机标识物...")
            left_markers = self.detect_markers(left_images[i])

            right_frame_basename = frame_basename.replace('lresult', 'rresult', 1)
            right_images_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_frame_basename)

            right_markers = self.detect_markers(right_images_path)
            print(f"  检测到 左: {len(left_markers)} 个, 右: {len(right_markers)} 个标识物。")

            print("步骤 2/4: 匹配与三维重建...")
            points_3d, matched_l, matched_r = self.match_and_triangulate(left_markers, right_markers)
            if points_3d.shape[0] < 10:
                print("  警告: 有效三维点过少 (<10)，无法进行拟合，跳过此帧。")
                continue
            print(f"  成功重建 {points_3d.shape[0]} 个稀疏三维点。")

            print("步骤 3/4: 拟合波浪表面并生成稠密视差图...")
            img_shape = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE).shape
            disparity_map, fit_params = self.fit_and_generate_dense_map(points_3d, img_shape)
            if disparity_map is None:
                print("  错误: 拟合失败，跳过此帧。")
                continue
            print("  成功生成稠密视差图。")

            if self.cfg.VISUALIZE_GT_GENERATION and (i % self.cfg.VISUALIZE_FRAME_INTERVAL == 0):
                print("步骤 4/4: 生成可视化结果...")
                self.visualize_results(left_images[i], right_images_path, left_markers, right_markers, points_3d,
                                       disparity_map, fit_params, matched_l, matched_r, frame_basename.split('.')[0])
                print("  可视化结果已保存。")

            with open(gt_path, 'wb') as f:
                pickle.dump(disparity_map, f)
            print(f"真值文件已保存至: {gt_path}")

    def visualize_results(self, l_path, r_path, l_markers, r_markers, points_3d, disp_map, fit_params, matched_l,
                          matched_r, basename):
        img_l_vis = cv2.imread(l_path)
        for marker in l_markers:
            cv2.circle(img_l_vis, (int(marker[0]), int(marker[1])), 10, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.cfg.VISUALIZATION_DIR, f"{basename}_01_markers_detected.png"), img_l_vis)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o', label='Reconstructed Points')
        ax.set_xlabel('X (mm)');
        ax.set_ylabel('Y (mm)');
        ax.set_zlabel('Z (mm)')
        ax.set_title(f"Frame {basename} - Sparse 3D Point Cloud")
        ax.legend()
        plt.savefig(os.path.join(self.cfg.VISUALIZATION_DIR, f"{basename}_02_sparse_3d.png"))
        plt.close(fig)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        x_min, x_max = np.min(points_3d[:, 0]), np.max(points_3d[:, 0])
        y_min, y_max = np.min(points_3d[:, 1]), np.max(points_3d[:, 1])
        x_surf = np.linspace(x_min, x_max, 50)
        y_surf = np.linspace(y_min, y_max, 50)
        xx_surf, yy_surf = np.meshgrid(x_surf, y_surf)
        zz_surf = self.wave_surface_func(np.vstack([xx_surf.ravel(), yy_surf.ravel()]), *fit_params)
        ax.plot_surface(xx_surf, yy_surf, zz_surf.reshape(xx_surf.shape), cmap='viridis', alpha=0.6)
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o', s=15, label='Original Points')
        ax.set_title(f"Frame {basename} - Fitted Wave Surface vs Original Points")
        ax.set_xlabel('X (mm)');
        ax.set_ylabel('Y (mm)');
        ax.set_zlabel('Z (mm)')
        ax.legend()
        plt.savefig(os.path.join(self.cfg.VISUALIZATION_DIR, f"{basename}_03_fitted_surface.png"))
        plt.close(fig)

        disp_vis = cv2.normalize(disp_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.cfg.VISUALIZATION_DIR, f"{basename}_04_disparity_map.png"), disp_vis)

        img_l = cv2.imread(l_path)
        img_r = cv2.imread(r_path)
        match_vis_img = np.concatenate((img_l, img_r), axis=1)
        for pt_l, pt_r in zip(matched_l, matched_r):
            x1, y1 = int(pt_l[0]), int(pt_l[1])
            x2, y2 = int(pt_r[0]) + img_l.shape[1], int(pt_r[1])
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(match_vis_img, (x1, y1), 5, color, -1)
            cv2.circle(match_vis_img, (x2, y2), 5, color, -1)
            cv2.line(match_vis_img, (x1, y1), (x2, y2), color, 1)
        cv2.imwrite(os.path.join(self.cfg.VISUALIZATION_DIR, f"{basename}_05_matches.png"), match_vis_img)


# --- 3. PyTorch Dataset ---
class WaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        left_img_path = self.left_images[frame_idx]
        frame_basename = os.path.basename(left_img_path)

        right_frame_basename = frame_basename.replace('lresult', 'rresult', 1)
        right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_frame_basename)
        gt_path = os.path.join(self.cfg.GROUND_TRUTH_DIR, f"gt_disparity_{frame_basename.split('.')[0]}.pkl")

        try:
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)
            if left_img is None or right_img is None:
                raise FileNotFoundError
            with open(gt_path, 'rb') as f:
                disparity_gt = pickle.load(f)
        except (FileNotFoundError, TypeError, IOError):
            return None

        h, w, _ = left_img.shape
        target_h, target_w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
        left_img = cv2.resize(left_img, (target_w, target_h))
        right_img = cv2.resize(right_img, (target_w, target_h))
        disparity_gt = cv2.resize(disparity_gt, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        disparity_gt *= (target_w / w)
        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0
        disparity_tensor = torch.from_numpy(disparity_gt).float().unsqueeze(0)
        return left_tensor, right_tensor, disparity_tensor


# --- 4. 深度学习模型 ---
class DINOv2StereoModel(nn.Module):
    def __init__(self, dino_model_name="facebook/dinov2-base", out_channels=1):
        super().__init__()
        print(f"加载DINOv2模型: {dino_model_name}...")
        self.dino = Dinov2Model.from_pretrained(dino_model_name)
        for param in self.dino.parameters():
            param.requires_grad = False
        feature_dim = 768
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        print("模型构建完成。")

    def forward(self, left_image, right_image):
        h, w = left_image.shape[-2:]
        with torch.no_grad():
            features_left = self.dino(left_image).last_hidden_state
            features_right = self.dino(right_image).last_hidden_state
        patch_h, patch_w = h // 14, w // 14
        features_left = features_left.permute(0, 2, 1).reshape(-1, 768, patch_h, patch_w)
        features_right = features_right.permute(0, 2, 1).reshape(-1, 768, patch_h, patch_w)
        x = torch.cat([features_left, features_right], dim=1)
        predicted_disparity = self.decoder(x)
        predicted_disparity = nn.functional.interpolate(
            predicted_disparity, size=(h, w), mode='bilinear', align_corners=True
        )
        return predicted_disparity


# --- 5. 训练循环 ---
def train(cfg: Config):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None, None)

    train_dataset = WaveStereoDataset(cfg, is_validation=False)
    val_dataset = WaveStereoDataset(cfg, is_validation=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                              num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    model = DINOv2StereoModel(dino_model_name=cfg.DINO_MODEL_NAME).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)
    print("--- 开始训练模型 ---")
    best_val_loss = float('inf')
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} [训练]")
        for data in pbar:
            if not data: continue
            left, right, gt_disp = data
            left, right, gt_disp = left.to(device), right.to(device), gt_disp.to(device)
            optimizer.zero_grad()
            pred_disp = model(left, right)
            mask = (gt_disp > 0) & (gt_disp < 1000)
            if torch.sum(mask) == 0: continue
            loss = criterion(pred_disp[mask], gt_disp[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} [验证]"):
                if not data: continue
                left, right, gt_disp = data
                left, right, gt_disp = left.to(device), right.to(device), gt_disp.to(device)
                pred_disp = model(left, right)
                mask = (gt_disp > 0) & (gt_disp < 1000)
                if torch.sum(mask) == 0: continue
                loss = criterion(pred_disp[mask], gt_disp[mask])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} -> 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"验证损失提升，模型已保存至: {save_path}")


# --- 6. 主执行模块 ---
if __name__ == '__main__':
    plt.switch_backend('Agg')
    config = Config()
    gt_generator = GroundTruthGenerator(config)
    gt_generator.process_all_frames()
    gt_files = glob.glob(os.path.join(config.GROUND_TRUTH_DIR, "*.pkl"))
    if len(gt_files) < 10:
        print(f"\n错误：生成的有效真值文件数量 ({len(gt_files)}) 过少，无法开始训练。")
        print("请检查可视化文件夹中的图像，确认标识物检测和表面拟合是否正常。")
    else:
        train(config)
