# dinov3_self_supervised_full.py
# 整合版：在你的原始脚本基础上修复 SSIM / PSNR，并加入验证可视化保存（保留 DINOv3 模型加载与数据流水线）

import os
import sys
import glob
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import argparse
import math
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- TensorBoard optional ---
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# --- transformers AutoModel for local DINOv3 load ---
try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80)
    print("【致命错误】: 无法从 'transformers' 库中导入 'AutoModel'。")
    print("请确保已安装 'transformers' 库: pip install transformers")
    print("同时建议安装 'accelerate' 以获得更好的性能: pip install accelerate")
    print("=" * 80)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(levelname)s: %(message)s")

# --- Config ---
PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
DATA_ROOT = os.path.dirname(PROJECT_ROOT)

@dataclass
class Config:
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")

    CHECKPOINT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints_self_supervised")
    VISUALIZATION_DIR: str = os.path.join(PROJECT_ROOT, "visualizations")
    LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
    TENSORBOARD_DIR: str = os.path.join(PROJECT_ROOT, "runs")

    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")

    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 100

    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 512
    MASK_THRESHOLD: int = 30

    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    MAX_DISPARITY: int = 128

    USE_MIXED_PRECISION: bool = True
    USE_DATA_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.5
    PHOTOMETRIC_LOSS_WEIGHTS: tuple = (0.85, 0.15)
    USE_CONSISTENCY_LOSS: bool = True
    CONSISTENCY_LOSS_WEIGHT: float = 0.1

    INITIAL_SMOOTHNESS_WEIGHT: float = 0.5
    SMOOTHNESS_WEIGHT_DECAY: float = 0.98

# --- SSIM 修复实现（返回相似度 0..1） ---
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        # 使用 math.exp 生成 float 列表，再转为 tensor（避免 torch.exp 对 float 报错）
        vals = [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
        gauss = torch.tensor(vals, dtype=torch.float32)
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D = self.gaussian(window_size, 1.5).unsqueeze(1)  # [W,1]
        _2D = _1D @ _1D.t()  # [W,W]
        w = _2D.unsqueeze(0).unsqueeze(0)  # [1,1,W,W]
        w = w.expand(channel, 1, window_size, window_size).contiguous()
        return w

    def forward(self, x, y):
        # x,y: [B,C,H,W] values assumed 0..1 (or same scale)
        _, channel, _, _ = x.size()
        if (self.window is None) or (self.channel != channel) or (self.window.device != x.device) or (self.window.dtype != x.dtype):
            window = self.create_window(self.window_size, channel).to(x.device).type(x.dtype)
            self.window = window
            self.channel = channel
        else:
            window = self.window.to(x.device).type(x.dtype)

        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=channel)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=channel)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x = F.conv2d(x * x, window, padding=self.window_size // 2, groups=channel) - mu_x_sq
        sigma_y = F.conv2d(y * y, window, padding=self.window_size // 2, groups=channel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=channel) - mu_xy

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        ssim_map = ssim_n / (ssim_d + 1e-8)
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.view(ssim_map.size(0), -1).mean(dim=1)

# --- 改进的自监督损失（保留掩码功能） ---
class ImprovedSelfSupervisedLoss(nn.Module):
    def __init__(self, smoothness_weight=0.1, photometric_weights=(0.85, 0.15),
                 use_consistency_loss=True, consistency_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.photometric_weights = photometric_weights
        self.use_consistency_loss = use_consistency_loss
        self.consistency_weight = consistency_weight
        self.ssim = SSIM()

    def forward(self, inputs, outputs):
        left_img = inputs["left_image"]
        right_img = inputs["right_image"]
        mask = inputs.get("mask", torch.ones_like(left_img[:, :1, :, :], device=left_img.device))
        pred_disp = outputs["disparity"]

        warped_right_image = self.inverse_warp(right_img, pred_disp)
        warped_left_image = self.inverse_warp(left_img, -pred_disp)

        mask_sum = mask.sum() + 1e-8

        # L1 Loss (masked)
        l1_loss_right_map = torch.abs(warped_right_image - left_img)
        l1_loss_right = (l1_loss_right_map * mask).sum() / mask_sum
        l1_loss_left_map = torch.abs(warped_left_image - right_img)
        l1_loss_left = (l1_loss_left_map * mask).sum() / mask_sum
        l1_loss = (l1_loss_right + l1_loss_left) / 2

        # SSIM Loss (masked) -> use 1 - ssim
        ssim_loss_right_map = 1.0 - self.ssim(warped_right_image, left_img)
        ssim_loss_right = (ssim_loss_right_map * mask).sum() / mask_sum
        ssim_loss_left_map = 1.0 - self.ssim(warped_left_image, right_img)
        ssim_loss_left = (ssim_loss_left_map * mask).sum() / mask_sum
        ssim_loss = (ssim_loss_right + ssim_loss_left) / 2

        photometric_loss = self.photometric_weights[0] * ssim_loss + self.photometric_weights[1] * l1_loss

        # Smoothness (global)
        smoothness_loss = self.compute_smoothness_loss(pred_disp, left_img)

        consistency_loss = 0.0
        if self.use_consistency_loss and "disparity_right" in outputs and outputs["disparity_right"] is not None:
            disp_left = outputs["disparity"]
            disp_right = outputs["disparity_right"]
            warped_disp_right = self.inverse_warp(disp_right, -disp_left)
            consistency_loss = (torch.abs(disp_left - warped_disp_right) * mask).sum() / mask_sum

        total_loss = photometric_loss + self.smoothness_weight * smoothness_loss
        if self.use_consistency_loss:
            total_loss = total_loss + self.consistency_weight * consistency_loss

        return {
            "total_loss": total_loss,
            "photometric_loss": photometric_loss,
            "smoothness_loss": smoothness_loss,
            "consistency_loss": consistency_loss,
            "warped_right_image": warped_right_image,
            "warped_left_image": warped_left_image
        }

    def inverse_warp(self, features, disp):
        B, C, H, W = features.shape
        # create meshgrid
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=features.device),
                                            torch.arange(W, device=features.device), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=0).float()  # [2,H,W]
        pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,H,W]
        disp = disp.squeeze(1)
        transformed_x = pixel_coords[:, 0, :, :] - disp
        grid = torch.stack([transformed_x, pixel_coords[:, 1, :, :]], dim=-1)
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def compute_smoothness_loss(self, disp, img):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_disp_x = grad_disp_x * torch.exp(-grad_img_x)
        grad_disp_y = grad_disp_y * torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()

# --- Dataset (as in your original script) ---
class RectifiedWaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.is_validation = is_validation

        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            print(f"【致命错误】在路径 '{cfg.LEFT_IMAGE_DIR}' 中没有找到任何图像文件。请检查文件结构。")
            sys.exit(1)

        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            self.map1_left = calib['map1_left']
            self.map2_left = calib['map2_left']
            self.map1_right = calib['map1_right']
            self.map2_right = calib['map2_right']
            self.roi_left = tuple(calib['roi_left'])
            self.roi_right = tuple(calib['roi_right'])
            logging.info(f"✓ 成功加载相机标定文件: {cfg.CALIBRATION_FILE}")
        except FileNotFoundError:
            logging.error(f"【致命错误】找不到标定文件: {cfg.CALIBRATION_FILE}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"【致命错误】加载标定文件失败: {e}")
            sys.exit(1)

        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = int(self.indices[idx])
        left_img_path = self.left_images[frame_idx]
        frame_basename = os.path.basename(left_img_path)

        if frame_basename.startswith('left'):
            right_frame_basename = 'right' + frame_basename[4:]
            right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_frame_basename)
        else:
            right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, frame_basename)

        if not os.path.exists(right_img_path):
            logging.warning(f"找不到对应的右图 '{right_img_path}'，返回 None 样本")
            return None

        left_img_raw = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        right_img_raw = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
        if left_img_raw is None or right_img_raw is None:
            return None

        left_rectified = cv2.remap(left_img_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

        x, y, w, h = self.roi_left
        left_rectified = left_rectified[y:y + h, x:x + w]
        x, y, w, h = self.roi_right
        right_rectified = right_rectified[y:y + h, x:x + w]

        target_h, target_w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
        left_img = cv2.resize(left_rectified, (target_w, target_h))
        right_img = cv2.resize(right_rectified, (target_w, target_h))

        # mask: based on left image brightness
        _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

        if not self.is_validation and self.cfg.USE_DATA_AUGMENTATION and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
            brightness_factor = 0.8 + 0.4 * np.random.rand()
            left_img_bgr = np.clip(left_img_bgr * brightness_factor, 0, 255).astype(np.uint8)
            right_img_bgr = np.clip(right_img_bgr * brightness_factor, 0, 255).astype(np.uint8)
            contrast_factor = 0.8 + 0.4 * np.random.rand()
            left_mean, right_mean = left_img_bgr.mean(), right_img_bgr.mean()
            left_img_bgr = np.clip((left_img_bgr - left_mean) * contrast_factor + left_mean, 0, 255).astype(np.uint8)
            right_img_bgr = np.clip((right_img_bgr - right_mean) * contrast_factor + right_mean, 0, 255).astype(np.uint8)

        left_tensor = torch.from_numpy(left_img_bgr.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img_bgr.transpose(2, 0, 1)).float() / 255.0

        return left_tensor, right_tensor, mask_tensor

# --- 3D hourglass and conv blocks (kept as in your original) ---
def conv_block_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class Hourglass3D(nn.Module):
    def __init__(self, in_channels):
        super(Hourglass3D, self).__init__()
        self.conv1a = conv_block_3d(in_channels, in_channels * 2, stride=2)
        self.conv1b = conv_block_3d(in_channels * 2, in_channels * 2)
        self.conv2a = conv_block_3d(in_channels * 2, in_channels * 4, stride=2)
        self.conv2b = conv_block_3d(in_channels * 4, in_channels * 4)
        self.deconv2 = conv_block_3d(in_channels * 4, in_channels * 2)
        self.deconv1 = conv_block_3d(in_channels * 2, in_channels)
        self.redir1 = conv_block_3d(in_channels * 2, in_channels * 2)
        self.redir0 = conv_block_3d(in_channels, in_channels)

    def forward(self, x):
        out_conv1 = self.conv1b(self.conv1a(x))
        out_conv2 = self.conv2b(self.conv2a(out_conv1))
        up2 = F.interpolate(out_conv2, size=out_conv1.shape[2:], mode='trilinear', align_corners=False)
        deconv2_out = self.deconv2(up2)
        deconv1_in = F.relu(deconv2_out + self.redir1(out_conv1), inplace=True)
        up1 = F.interpolate(deconv1_in, size=x.shape[2:], mode='trilinear', align_corners=False)
        deconv1_out = self.deconv1(up1)
        final_out = F.relu(deconv1_out + self.redir0(x), inplace=True)
        return final_out

class DINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_disp = cfg.MAX_DISPARITY
        self.dino = self._load_dino_model()
        if self.dino is None:
            sys.exit(1)

        for param in self.dino.parameters():
            param.requires_grad = False

        # get config params from dino if available
        self.feature_dim = getattr(self.dino.config, 'hidden_size', 768)
        self.patch_size = getattr(self.dino.config, 'patch_size', 8)
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

        self.cost_aggregator = nn.Sequential(
            conv_block_3d(self.feature_dim, 32),
            Hourglass3D(32),
            nn.Conv3d(32, 1, 3, padding=1)
        )
        logging.info("✓ 模型构建完成 (3D沙漏网络架构)。")

    def _load_dino_model(self):
        local_path = self.cfg.DINO_LOCAL_PATH
        logging.info(f"--- 正在尝试从本地路径加载DINOv3模型: {local_path} ---")
        if not os.path.isdir(local_path) or not os.listdir(local_path):
            logging.error(f"【致命错误】: 在指定的本地路径中找不到模型文件: {local_path}")
            return None
        try:
            model = AutoModel.from_pretrained(local_path, local_files_only=True)
            logging.info("✓ 成功从本地加载DINOv3模型。")
            return model
        except Exception as e:
            logging.error(f"【致命错误】: 从本地路径 '{local_path}' 加载模型时发生错误: {e}")
            return None

    def get_features(self, image):
        b, c, h, w = image.shape
        with torch.no_grad():
            outputs = self.dino(image)
            features = outputs.last_hidden_state
        start_index = 1 + self.num_register_tokens
        patch_tokens = features[:, start_index:, :]
        feature_h, feature_w = h // self.patch_size, w // self.patch_size
        features_2d = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, feature_h, feature_w)
        return features_2d

    def build_cost_volume(self, left_feat, right_feat):
        B, C, H, W = left_feat.shape
        max_disp_feat = self.max_disp // self.patch_size
        cost_volume = torch.zeros(B, C, max_disp_feat, H, W, device=left_feat.device, dtype=left_feat.dtype)
        for d in range(max_disp_feat):
            if d > 0:
                cost_volume[:, :, d, :, d:] = left_feat[:, :, :, d:] - right_feat[:, :, :, :-d]
            else:
                cost_volume[:, :, d, :, :] = left_feat - right_feat
        return cost_volume

    def forward(self, left_image, right_image):
        h, w = left_image.shape[-2:]
        left_feat = self.get_features(left_image)
        right_feat = self.get_features(right_image)
        cost_volume = self.build_cost_volume(left_feat, right_feat)
        cost_aggregated = self.cost_aggregator(cost_volume).squeeze(1)
        cost_softmax = F.softmax(-cost_aggregated, dim=1)
        max_disp_feat = self.max_disp // self.patch_size
        disp_values = torch.arange(0, max_disp_feat, device=cost_softmax.device, dtype=torch.float32).view(1, -1, 1, 1)
        disparity_feat = torch.sum(cost_softmax * disp_values, 1, keepdim=True)
        disparity = F.interpolate(disparity_feat * self.patch_size, size=(h, w), mode='bilinear', align_corners=False)
        disparity_right = None
        if self.cfg.USE_CONSISTENCY_LOSS:
            cost_volume_right = self.build_cost_volume(right_feat, left_feat)
            cost_aggregated_right = self.cost_aggregator(cost_volume_right).squeeze(1)
            cost_softmax_right = F.softmax(-cost_aggregated_right, dim=1)
            disparity_feat_right = torch.sum(cost_softmax_right * disp_values, 1, keepdim=True)
            disparity_right = F.interpolate(disparity_feat_right * self.patch_size, size=(h, w), mode='bilinear', align_corners=False)
        return {"disparity": disparity, "disparity_right": disparity_right}

# --- Evaluation metrics（PSNR 修复） ---
class EvaluationMetrics:
    @staticmethod
    def compute_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return (20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))).item()

    @staticmethod
    def compute_rmse(img1, img2):
        return torch.sqrt(torch.mean((img1 - img2) ** 2)).item()

    @staticmethod
    def compute_ssim(img1, img2):
        ssim_module = SSIM()
        return ssim_module(img1, img2).item()

    @staticmethod
    def evaluate_reconstruction(inputs, loss_components):
        left_img = inputs["left_image"]
        warped_right = loss_components["warped_right_image"]
        return {
            "psnr": EvaluationMetrics.compute_psnr(left_img, warped_right),
            "rmse": EvaluationMetrics.compute_rmse(left_img, warped_right),
            "ssim": EvaluationMetrics.compute_ssim(left_img, warped_right),
        }

# --- Trainer / Utilities ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        os.makedirs(cfg.TENSORBOARD_DIR, exist_ok=True)
        if cfg.VISUALIZE_TRAINING:
            os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"✓ 使用设备: {self.device}")

        self.writer = None
        if SummaryWriter:
            self.writer = SummaryWriter(log_dir=os.path.join(cfg.TENSORBOARD_DIR, datetime.now().strftime('%Y%m%d-%H%M%S')))

        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.loss_fn = ImprovedSelfSupervisedLoss(
            smoothness_weight=cfg.INITIAL_SMOOTHNESS_WEIGHT,
            photometric_weights=cfg.PHOTOMETRIC_LOSS_WEIGHTS,
            use_consistency_loss=cfg.USE_CONSISTENCY_LOSS,
            consistency_weight=cfg.CONSISTENCY_LOSS_WEIGHT
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-6)
        self.evaluator = EvaluationMetrics()

        train_dataset = RectifiedWaveStereoDataset(cfg, is_validation=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=self.device.type == 'cuda')
        val_dataset = RectifiedWaveStereoDataset(cfg, is_validation=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=self.device.type == 'cuda')

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_MIXED_PRECISION and self.device.type == 'cuda')
        self.step = 0
        self.current_smoothness_weight = cfg.INITIAL_SMOOTHNESS_WEIGHT

        self.loss_history = {'train': {'total': [], 'photometric': [], 'smoothness': [], 'consistency': []},
                             'val': {'total': [], 'photometric': [], 'smoothness': [], 'consistency': []}}
        self.metric_history = {'train': {'psnr': [], 'rmse': [], 'ssim': []},
                               'val': {'psnr': [], 'rmse': [], 'ssim': []}}

        self._setup_visualization_font()
        self.log_file = os.path.join(cfg.LOG_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def _setup_visualization_font(self):
        font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'sans-serif']
        for font_name in font_names:
            try:
                if any(font.name == font_name for font in fm.fontManager.ttflist):
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    logging.info(f"✓ 可视化已配置中文字体: {font_name}")
                    return
            except Exception:
                continue
        logging.warning("未找到指定中文字体，可视化标题可能显示异常。")

    def train(self):
        logging.info("\n--- 开始自监督训练 ---")
        best_val_loss = float('inf')

        for epoch in range(self.cfg.NUM_EPOCHS):
            self.current_smoothness_weight *= self.cfg.SMOOTHNESS_WEIGHT_DECAY
            self.loss_fn.smoothness_weight = self.current_smoothness_weight

            self.model.train()
            epoch_train_loss_total = 0.0
            epoch_train_metrics = {'psnr': 0.0, 'rmse': 0.0, 'ssim': 0.0}
            last_train_loss_components = {}

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [训练]")

            for data in pbar:
                if data is None or data[0] is None:
                    continue
                left, right, mask = [d.to(self.device) for d in data]
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=(self.cfg.USE_MIXED_PRECISION and self.device.type == 'cuda')):
                    outputs = self.model(left, right)
                    inputs = {"left_image": left, "right_image": right, "mask": mask}
                    loss_components = self.loss_fn(inputs, outputs)
                    loss = loss_components["total_loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"训练步骤 {self.step} 出现无效损失 (NaN/inf)，跳过。")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.cfg.GRADIENT_CLIP_VAL)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                step_metrics = self.evaluator.evaluate_reconstruction(inputs, loss_components)
                epoch_train_loss_total += loss.item()
                for k in epoch_train_metrics:
                    epoch_train_metrics[k] += step_metrics[k]
                last_train_loss_components = loss_components

                pbar.set_postfix({'loss': loss.item(), 'psnr': step_metrics['psnr'], 'lr': self.optimizer.param_groups[0]['lr']})

                if self.writer:
                    self.writer.add_scalar('Loss/train_step', loss.item(), self.step)
                    self.writer.add_scalar('Metrics/train_psnr_step', step_metrics['psnr'], self.step)
                    self.writer.add_scalar('Params/learning_rate', self.optimizer.param_groups[0]['lr'], self.step)

                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(inputs, outputs, loss_components, self.step, "train")

                self.step += 1

            train_len = len(self.train_loader)
            if train_len > 0:
                avg_train_loss = epoch_train_loss_total / train_len
                avg_train_metrics = {k: v / train_len for k, v in epoch_train_metrics.items()}
                self.loss_history['train']['total'].append(avg_train_loss)
                self.loss_history['train']['photometric'].append(last_train_loss_components.get('photometric_loss', torch.tensor(0)).item())
                self.loss_history['train']['smoothness'].append(last_train_loss_components.get('smoothness_loss', torch.tensor(0)).item())
                self.loss_history['train']['consistency'].append(last_train_loss_components.get('consistency_loss', torch.tensor(0)).item())
                for k, v in avg_train_metrics.items():
                    self.metric_history['train'][k].append(v)
                if self.writer:
                    self.writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
                    self.writer.add_scalar('Metrics/train_psnr_epoch', avg_train_metrics['psnr'], epoch)

            avg_val_loss, val_metrics = self.validate(epoch)
            logging.info(f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> 验证损失: {avg_val_loss:.4f}, PSNR: {val_metrics.get('psnr', 0):.2f}")

            self.scheduler.step()
            self.update_log_file(epoch)
            if self.cfg.VISUALIZE_TRAINING:
                self.plot_training_history()

            if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model_self_supervised.pth")
                torch.save(self.model.state_dict(), save_path)
                logging.info(f"✓ 验证损失降低，模型已保存至: {save_path}")

        logging.info("训练完成!")
        if self.writer: self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        total_loss, val_psnr, val_ssim, val_rmse = 0.0, 0.0, 0.0, 0.0
        last_val_loss_components = {}

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_loader, desc="[验证]")):
                if data is None or data[0] is None:
                    continue
                left, right, mask = [d.to(self.device) for d in data]
                outputs = self.model(left, right)
                inputs = {"left_image": left, "right_image": right, "mask": mask}
                loss_components = self.loss_fn(inputs, outputs)
                last_val_loss_components = loss_components

                if not torch.isnan(loss_components["total_loss"]):
                    total_loss += loss_components["total_loss"].item()
                step_metrics = self.evaluator.evaluate_reconstruction(inputs, loss_components)
                val_psnr += step_metrics['psnr']
                val_ssim += step_metrics['ssim']
                val_rmse += step_metrics['rmse']

                # save first batch visualization during validation
                if i == 0 and self.cfg.VISUALIZE_TRAINING:
                    self.visualize(inputs, outputs, loss_components, epoch, "val")

        val_len = len(self.val_loader)
        avg_loss = total_loss / val_len if val_len > 0 else float('nan')
        avg_metrics = {'psnr': val_psnr / val_len, 'ssim': val_ssim / val_len, 'rmse': val_rmse / val_len} if val_len > 0 else {}

        if val_len > 0:
            self.loss_history['val']['total'].append(avg_loss)
            self.loss_history['val']['photometric'].append(last_val_loss_components.get('photometric_loss', torch.tensor(0)).item())
            self.loss_history['val']['smoothness'].append(last_val_loss_components.get('smoothness_loss', torch.tensor(0)).item())
            self.loss_history['val']['consistency'].append(last_val_loss_components.get('consistency_loss', torch.tensor(0)).item())
            for k, v in avg_metrics.items():
                self.metric_history['val'][k].append(v)
            if self.writer and self.loss_history['train']['total']:
                self.writer.add_scalar('Loss/validation_epoch', avg_loss, epoch)
                self.writer.add_scalar('Metrics/validation_psnr_epoch', avg_metrics['psnr'], epoch)
                self.writer.add_scalars('Loss/epoch_comparison',
                                        {'train': self.loss_history['train']['total'][-1], 'validation': avg_loss},
                                        epoch)

        return avg_loss, avg_metrics

    def visualize(self, inputs, outputs, loss_components, step, phase="train"):
        left_img = inputs["left_image"][0].permute(1, 2, 0).cpu().numpy()
        pred_disp = outputs["disparity"][0, 0].cpu().detach().numpy()
        mask = inputs["mask"][0, 0].cpu().numpy()

        fig = plt.figure(figsize=(12, 12))
        plt.suptitle(f'可视化 - 步骤: {step} ({phase})', fontsize=16)

        plt.subplot(3, 2, 1)
        plt.imshow(left_img)
        plt.title("校正后的左图 (输入)")
        plt.axis('off')

        ax2 = plt.subplot(3, 2, 2)
        im2 = ax2.imshow(pred_disp, cmap='viridis')
        plt.title("预测视差图")
        plt.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        warped_right = loss_components["warped_right_image"][0].permute(1, 2, 0).cpu().detach().numpy()
        plt.subplot(3, 2, 3)
        plt.imshow(warped_right)
        plt.title("重建的左图 (来自右图+视差)")
        plt.axis('off')

        diff_right = np.clip(np.abs(left_img - warped_right), 0, 1)
        ax4 = plt.subplot(3, 2, 4)
        im4 = ax4.imshow(diff_right.mean(axis=2), cmap='hot')
        plt.title("光度误差图")
        plt.axis('off')
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        plt.subplot(3, 2, 5)
        plt.imshow(mask, cmap='gray')
        plt.title("注意力掩码 (仅在此区域计算光度损失)")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if self.writer:
            self.writer.add_figure(f'Visualization/{phase}', fig, global_step=step)

        fig.canvas.draw()
        img_buffer = fig.canvas.tostring_argb()
        img_np = np.frombuffer(img_buffer, dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_np = img_np.reshape(h, w, 4)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, f"{phase}_step_{step:06d}.png")
        cv2.imwrite(save_path, img_bgr)
        plt.close(fig)

    def plot_training_history(self):
        if not self.loss_history['train']['total']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练历史曲线', fontsize=16)

        axes[0, 0].plot(self.loss_history['train']['total'], label='训练损失')
        if self.loss_history['val']['total']:
            axes[0, 0].plot(self.loss_history['val']['total'], label='验证损失')
        axes[0, 0].set_title('总损失'); axes[0, 0].legend(); axes[0, 0].grid(True)

        axes[0, 1].plot(self.loss_history['train']['photometric'], label='光度损失')
        axes[0, 1].plot(self.loss_history['train']['smoothness'], label='平滑损失')
        axes[0, 1].set_title('损失组件'); axes[0, 1].legend(); axes[0, 1].grid(True)

        if self.metric_history['train']['psnr']:
            axes[1, 0].plot(self.metric_history['train']['psnr'], label='训练PSNR')
        if self.metric_history['val']['psnr']:
            axes[1, 0].plot(self.metric_history['val']['psnr'], label='验证PSNR')
        axes[1, 0].set_title('PSNR'); axes[1, 0].legend(); axes[1, 0].grid(True)

        if self.metric_history['train']['ssim']:
            axes[1, 1].plot(self.metric_history['train']['ssim'], label='训练SSIM')
        if self.metric_history['val']['ssim']:
            axes[1, 1].plot(self.metric_history['val']['ssim'], label='验证SSIM')
        axes[1, 1].set_title('SSIM'); axes[1, 1].legend(); axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.canvas.draw()
        img_buffer = fig.canvas.tostring_argb()
        img_np = np.frombuffer(img_buffer, dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_np = img_np.reshape(h, w, 4)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, "training_history.png")
        cv2.imwrite(save_path, img_bgr)
        plt.close(fig)

    def update_log_file(self, epoch):
        log_data = {
            'config': {k: str(v) for k, v in asdict(self.cfg).items()},
            'epoch': epoch,
            'loss_history': self.loss_history,
            'metric_history': self.metric_history,
            'update_time': datetime.now().isoformat()
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

# --- 主执行 (保留你的入口约定) ---
if __name__ == '__main__':
    # 检查 TensorBoard
    if SummaryWriter is None:
        logging.warning("TensorBoard 未安装：TensorBoard 日志将不可用 (可选)。")

    plt.switch_backend('Agg')

    config = Config()
    trainer = Trainer(config)
    trainer.train()
