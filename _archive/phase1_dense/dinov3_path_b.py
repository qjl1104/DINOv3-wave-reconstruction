# dinov3_path_b.py
# 实现了从含泡沫图像直接重建无泡沫波浪表面的端到端深度学习方案
# (版本已根据用户要求升级为DINOv3)

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

# 确保可以导入transformers库, 如果没有请运行: pip install transformers torch
try:
    # --- 已更新为DINOv3 ---
    from transformers import Dinov3Model
except ImportError:
    print("错误: 'transformers' 库未找到。请运行 'pip install transformers' 进行安装。")
    exit()


# --- 1. 配置中心 ---
@dataclass
class Config:
    """项目配置参数"""
    # --- 文件路径 (已根据您的环境配置) ---
    # !! 注意: Python中建议使用正斜杠'/'作为路径分隔符, 在Windows上同样有效 !!
    LEFT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/lresult/"
    RIGHT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/rresult/"
    CALIBRATION_FILE: str = "D:/Research/wave_reconstruction_project/camera_calibration/params/stereo_calib_params_from_matlab_full.npz"

    # --- 自动生成的数据路径 (相对于脚本位置) ---
    # 真值视差图将保存在: D:/Research/wave_reconstruction_project/data/ground_truth_disparity/
    GROUND_TRUTH_DIR: str = "D:/Research/wave_reconstruction_project/data/ground_truth_disparity/"
    # 模型 checkpoints 将保存在: D:/Research/wave_reconstruction_project/DINOv3/checkpoints/
    CHECKPOINT_DIR: str = "./checkpoints/"

    # --- 数据处理参数 ---
    IMAGE_HEIGHT: int = 320  # 调整输入图像尺寸以适应模型和显存
    IMAGE_WIDTH: int = 512

    # --- 模型与训练参数 ---
    # --- 已更新为DINOv3 ---
    DINO_MODEL_NAME: str = "facebook/dinov3_vitb14"
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1  # 10%的数据用于验证


# --- 2. 真值生成器 (借鉴您的传统CV代码) ---
class GroundTruthGenerator:
    """
    自动化生成稠密、干净的视差图真值。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.GROUND_TRUTH_DIR, exist_ok=True)

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
        """从单张图像中检测标识物中心点。"""
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

    def match_and_triangulate(self, left_markers: np.ndarray, right_markers: np.ndarray) -> np.ndarray:
        """匹配左右标识物并进行三维三角化。"""
        if left_markers.shape[0] == 0 or right_markers.shape[0] == 0:
            return np.array([])

        cost_matrix = np.full((len(left_markers), len(right_markers)), np.inf)
        for i, p_l in enumerate(left_markers):
            for j, p_r in enumerate(right_markers):
                if abs(p_l[1] - p_r[1]) < 10 and p_l[0] > p_r[0]:
                    cost_matrix[i, j] = np.linalg.norm(p_l - p_r)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_points_l, matched_points_r = [], []
        for r, c in zip(row_ind, col_ind):
            if np.isfinite(cost_matrix[r, c]):
                matched_points_l.append(left_markers[r])
                matched_points_r.append(right_markers[c])

        if not matched_points_l: return np.array([])

        points_l_np = np.array(matched_points_l).T
        points_r_np = np.array(matched_points_r).T

        points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, points_l_np, points_r_np)
        points_3d = (points_4d_hom[:3] / (points_4d_hom[3] + 1e-8)).T
        return points_3d

    @staticmethod
    def wave_surface_func(xy_data, amplitude, freq_x, phase_x, freq_y, phase_y, offset):
        """二维正弦波函数模型"""
        x, y = xy_data[0, :], xy_data[1, :]
        z = amplitude * np.cos(freq_x * x + phase_x) * np.cos(freq_y * y + phase_y) + offset
        return z

    def fit_and_generate_dense_map(self, points_3d: np.ndarray, orig_shape: tuple) -> np.ndarray:
        """从稀疏3D点拟合波浪表面，并生成稠密视差图"""
        if len(points_3d) < 10: return None

        x_data, y_data, z_data = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        try:
            initial_guess = [np.std(z_data), 0.1, 0, 0.1, 0, np.mean(z_data)]
            params, _ = curve_fit(self.wave_surface_func, np.vstack((x_data, y_data)), z_data, p0=initial_guess,
                                  maxfev=5000)
        except RuntimeError:
            print("警告: 波浪表面拟合失败。")
            return None

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
        return disparity_map.astype(np.float32)

    def process_all_frames(self):
        """主流程：处理所有帧以生成或加载真值数据"""
        print("--- 开始生成或检查真值数据 ---")
        left_images = sorted(glob.glob(os.path.join(self.cfg.LEFT_IMAGE_DIR, "*.*")))
        right_images = sorted(glob.glob(os.path.join(self.cfg.RIGHT_IMAGE_DIR, "*.*")))

        if len(left_images) != len(right_images) or not left_images:
            print(f"错误: 左右相机图像数量不匹配或未找到图像。左: {len(left_images)}, 右: {len(right_images)}")
            return

        for i in tqdm(range(len(left_images)), desc="生成真值"):
            frame_basename = os.path.basename(left_images[i]).split('.')[0]
            gt_path = os.path.join(self.cfg.GROUND_TRUTH_DIR, f"gt_disparity_{frame_basename}.pkl")
            if os.path.exists(gt_path): continue

            left_markers = self.detect_markers(left_images[i])
            right_markers = self.detect_markers(right_images[i])

            points_3d = self.match_and_triangulate(left_markers, right_markers)

            if points_3d.shape[0] > 10:
                img_shape = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE).shape
                disparity_map = self.fit_and_generate_dense_map(points_3d, img_shape)

                if disparity_map is not None:
                    with open(gt_path, 'wb') as f:
                        pickle.dump(disparity_map, f)


# --- 3. PyTorch Dataset ---
class WaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))

        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42)  # for reproducibility
        np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))

        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        left_img_path = self.left_images[frame_idx]

        frame_basename = os.path.basename(left_img_path)
        right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, frame_basename)
        gt_path = os.path.join(self.cfg.GROUND_TRUTH_DIR, f"gt_disparity_{frame_basename.split('.')[0]}.pkl")

        try:
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)
            with open(gt_path, 'rb') as f:
                disparity_gt = pickle.load(f)
        except (FileNotFoundError, TypeError, IOError) as e:
            print(f"警告: 无法加载样本 {frame_idx} (文件: {gt_path}). 原因: {e}. 将跳过。")
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
class DINOv3StereoModel(nn.Module):
    def __init__(self, dino_model_name="facebook/dinov3_vitb14", out_channels=1):
        super().__init__()
        print(f"加载DINOv3模型: {dino_model_name}...")
        # --- 已更新为DINOv3 ---
        self.dino = Dinov3Model.from_pretrained(dino_model_name)
        for param in self.dino.parameters():
            param.requires_grad = False

        # DINOv3 ViT-B/14 的特征维度是 768
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
            nn.ReLU()  # 视差总是正数
        )
        print("模型构建完成。")

    def forward(self, left_image, right_image):
        h, w = left_image.shape[-2:]

        with torch.no_grad():
            features_left = self.dino(left_image).last_hidden_state
            features_right = self.dino(right_image).last_hidden_state

        # DINOv3 ViT-B/14 的 patch size 是 14
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

    model = DINOv3StereoModel(dino_model_name=cfg.DINO_MODEL_NAME).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)

    print("--- 开始训练模型 ---")
    best_val_loss = float('inf')

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} [训练]")
        for left, right, gt_disp in pbar:
            if not left: continue

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
            for left, right, gt_disp in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} [验证]"):
                if not left: continue
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
    config = Config()

    gt_generator = GroundTruthGenerator(config)
    gt_generator.process_all_frames()

    train(config)
