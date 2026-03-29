# dinov3_full_integrated.py
"""
完整整合版：DINOv3 特征 + 3D hourglass cost-aggregator 自监督双目深度/视差训练脚本
- 包含：数据集、DINOv3 本地/Hub 加载、特征提取、cost volume、3D 沙漏、损失(SSIM+L1+smoothness+consistency)、评估、可视化、训练循环
- AMP 正确使用：amp.autocast(device_type=...), GradScaler
- 可通过命令行覆盖左右图像目录与校准文件
"""

import os
import sys
import glob
import json
import math
import argparse
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple

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

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# Optional transformers (DINO)
try:
    from transformers import AutoModel
except Exception:
    AutoModel = None

# AMP modern API
from torch import amp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    PROJECT_ROOT: str = r"D:\Research\wave_reconstruction_project\DINOv3"
    DATA_ROOT: str = os.path.dirname(PROJECT_ROOT)

    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")

    CHECKPOINT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints_self_supervised")
    VISUALIZATION_DIR: str = os.path.join(PROJECT_ROOT, "visualizations")
    LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
    TENSORBOARD_DIR: str = os.path.join(PROJECT_ROOT, "runs")

    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")
    DINO_ONLINE_MODEL: str = "facebook/dinov3_vitb14"

    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 512
    BATCH_SIZE: int = 4
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    MAX_DISPARITY: int = 128

    USE_DATA_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.5
    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 200

    PHOTOMETRIC_LOSS_WEIGHTS: Tuple[float, float] = (0.85, 0.15)
    USE_CONSISTENCY_LOSS: bool = True
    CONSISTENCY_LOSS_WEIGHT: float = 0.1
    INITIAL_SMOOTHNESS_WEIGHT: float = 0.5
    SMOOTHNESS_WEIGHT_DECAY: float = 0.98

    USE_MIXED_PRECISION: bool = True
    PSNR_MAX_VAL: float = 1.0

# -------------------------
# Path resolver
# -------------------------
def resolve_image_dir(default_path: str, name: str, project_root: str) -> str:
    env_key = f"{name.upper()}_IMAGE_DIR"
    if os.environ.get(env_key):
        p = os.environ[env_key]
        logging.info(f"Using {env_key} from environment: {p}")
        return p

    candidates = [default_path,
                  os.path.join(project_root, "data", os.path.basename(default_path)),
                  os.path.join(os.path.dirname(project_root), "data", os.path.basename(default_path)),
                  os.path.join(os.path.dirname(os.path.dirname(project_root)), "data", os.path.basename(default_path))]
    candidates = [os.path.normpath(c) for c in candidates if c]
    seen = set(); uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c); seen.add(c)
    for c in uniq:
        if os.path.isdir(c):
            imgs = glob.glob(os.path.join(c, "*.*"))
            if imgs:
                logging.info(f"Resolved {name} dir to: {c} (found {len(imgs)} files)")
                return c
    logging.warning(f"Could not auto-resolve '{name}' dir. Tried:\n  " + "\n  ".join(uniq))
    logging.warning(f"Will use default: {default_path}")
    return default_path

# -------------------------
# SSIM implementation
# -------------------------
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.register_buffer('window', self._create_window(window_size, 1))

    def _gaussian(self, window_size, sigma):
        vals = [math.exp(-((x - window_size // 2) ** 2) / (2 * sigma * sigma)) for x in range(window_size)]
        gauss = torch.tensor(vals, dtype=torch.float32)
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D @ _1D.t()
        w = _2D.unsqueeze(0).unsqueeze(0)
        w = w.expand(channel, 1, window_size, window_size).contiguous()
        return w

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if x.shape != y.shape:
            raise ValueError("SSIM inputs must have same shape")
        B, C, H, W = x.shape
        if (self.window.shape[0] != C) or (self.window.device != x.device) or (self.window.dtype != x.dtype):
            window = self._create_window(self.window_size, C).to(x.device).type(x.dtype)
            self.register_buffer('window', window)
        window = self.window
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=C)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(x * x, window, padding=self.window_size // 2, groups=C) - mu_x_sq
        sigma_y = F.conv2d(y * y, window, padding=self.window_size // 2, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=C) - mu_xy
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        ssim_map = ssim_n / (ssim_d + 1e-8)
        ssim_map = torch.clamp((1.0 + ssim_map) / 2.0, 0.0, 1.0)
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.view(B, -1).mean(dim=1)

# -------------------------
# Dataset
# -------------------------
class RectifiedWaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, left_dir: str, right_dir: str, is_validation: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_validation = is_validation
        self.left_dir = left_dir
        self.right_dir = right_dir

        self.left_images = sorted(glob.glob(os.path.join(left_dir, "*.*")))
        if not self.left_images:
            logging.error(f"No left images found in {left_dir}")
            raise FileNotFoundError(left_dir)

        # load calibration if exists
        self.map1_left = self.map2_left = self.map1_right = self.map2_right = None
        self.roi_left = self.roi_right = None
        if os.path.exists(cfg.CALIBRATION_FILE):
            try:
                calib = np.load(cfg.CALIBRATION_FILE)
                self.map1_left = calib.get('map1_left', None)
                self.map2_left = calib.get('map2_left', None)
                self.map1_right = calib.get('map1_right', None)
                self.map2_right = calib.get('map2_right', None)
                self.roi_left = tuple(calib.get('roi_left', (0, 0, -1, -1)))
                self.roi_right = tuple(calib.get('roi_right', (0, 0, -1, -1)))
                logging.info(f"Loaded calibration from {cfg.CALIBRATION_FILE}")
            except Exception as e:
                logging.warning(f"Failed to load calibration: {e}")
        else:
            logging.warning(f"Calibration file not found: {cfg.CALIBRATION_FILE}. Proceeding without remap.")

        num = len(self.left_images)
        indices = np.arange(num)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(num * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = int(self.indices[idx])
        left_path = self.left_images[frame_idx]
        basename = os.path.basename(left_path)

        if basename.startswith('left'):
            right_name = 'right' + basename[4:]
        else:
            right_name = basename
        right_path = os.path.join(self.right_dir, right_name)
        if not os.path.exists(right_path):
            right_path = os.path.join(self.right_dir, basename)
            if not os.path.exists(right_path):
                logging.warning(f"Right image not found for {left_path}, skipping.")
                return None

        left_raw = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_raw = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if left_raw is None or right_raw is None:
            logging.warning(f"Failed to read images: {left_path} or {right_path}")
            return None

        # remap
        if self.map1_left is not None and self.map2_left is not None:
            left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        else:
            left_rect = left_raw
        if self.map1_right is not None and self.map2_right is not None:
            right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        else:
            right_rect = right_raw

        # crop ROI if given
        if self.roi_left and len(self.roi_left) == 4 and self.roi_left[2] > 0:
            x, y, w, h = self.roi_left
            left_rect = left_rect[y:y+h, x:x+w]
        if self.roi_right and len(self.roi_right) == 4 and self.roi_right[2] > 0:
            x, y, w, h = self.roi_right
            right_rect = right_rect[y:y+h, x:x+w]

        th, tw = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
        left_img = cv2.resize(left_rect, (tw, th))
        right_img = cv2.resize(right_rect, (tw, th))

        _, mask = cv2.threshold(left_img, 30, 255, cv2.THRESH_BINARY)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0  # [1,H,W]

        left_bgr = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

        if (not self.is_validation) and self.cfg.USE_DATA_AUGMENTATION and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
            bf = 0.8 + 0.4 * np.random.rand()
            cf = 0.8 + 0.4 * np.random.rand()
            lm, rm = left_bgr.mean(), right_bgr.mean()
            left_bgr = np.clip(left_bgr * bf, 0, 255).astype(np.uint8)
            right_bgr = np.clip(right_bgr * bf, 0, 255).astype(np.uint8)
            left_bgr = np.clip((left_bgr - lm) * cf + lm, 0, 255).astype(np.uint8)
            right_bgr = np.clip((right_bgr - rm) * cf + rm, 0, 255).astype(np.uint8)

        left_t = torch.from_numpy(left_bgr.transpose(2,0,1)).float() / 255.0
        right_t = torch.from_numpy(right_bgr.transpose(2,0,1)).float() / 255.0

        return left_t, right_t, mask_tensor

# -------------------------
# Model (3D hourglass cost-aggregator)
# -------------------------
def conv_block_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class Hourglass3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1a = conv_block_3d(in_channels, in_channels*2, stride=2)
        self.conv1b = conv_block_3d(in_channels*2, in_channels*2)
        self.conv2a = conv_block_3d(in_channels*2, in_channels*4, stride=2)
        self.conv2b = conv_block_3d(in_channels*4, in_channels*4)
        self.deconv2 = conv_block_3d(in_channels*4, in_channels*2)
        self.deconv1 = conv_block_3d(in_channels*2, in_channels)
        self.redir1 = conv_block_3d(in_channels*2, in_channels*2)
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
        self.dino = self._load_dino()
        if self.dino is None:
            raise RuntimeError("Failed to load DINO model.")
        for p in self.dino.parameters():
            p.requires_grad = False

        self.feature_dim = getattr(self.dino.config, 'hidden_size', 768)
        self.patch_size = getattr(self.dino.config, 'patch_size', 8)
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

        self.cost_aggregator = nn.Sequential(
            conv_block_3d(self.feature_dim, 32),
            Hourglass3D(32),
            nn.Conv3d(32, 1, 3, padding=1)
        )

        logging.info(f"Model constructed: feature_dim={self.feature_dim}, patch_size={self.patch_size}")

    def _load_dino(self):
        if AutoModel is None:
            logging.error("transformers.AutoModel not available. Install transformers.")
            return None
        local = self.cfg.DINO_LOCAL_PATH
        try:
            if os.path.isdir(local) and os.listdir(local):
                logging.info(f"Loading DINO from local: {local}")
                return AutoModel.from_pretrained(local, local_files_only=True)
        except Exception as e:
            logging.warning(f"Local DINO load failed: {e}")
        try:
            logging.info(f"Loading DINO from hub: {self.cfg.DINO_ONLINE_MODEL}")
            model = AutoModel.from_pretrained(self.cfg.DINO_ONLINE_MODEL)
            try:
                model.save_pretrained(local)
                logging.info(f"Saved DINO to local: {local}")
            except Exception:
                logging.debug("Failed to save DINO locally.")
            return model
        except Exception as e:
            logging.error(f"Failed to load DINO from hub: {e}")
            return None

    def get_features(self, image: torch.Tensor):
        b, c, h, w = image.shape
        with torch.no_grad():
            outputs = self.dino(image)
            features = getattr(outputs, 'last_hidden_state', None)
            if features is None:
                raise RuntimeError("DINO outputs missing last_hidden_state")
        start = 1 + self.num_register_tokens
        patch_tokens = features[:, start:, :]
        ph = h // self.patch_size
        pw = w // self.patch_size
        feat2d = patch_tokens.permute(0,2,1).reshape(b, self.feature_dim, ph, pw)
        return feat2d

    def build_cost_volume(self, left_feat, right_feat):
        B, C, H, W = left_feat.shape
        max_disp_feat = max(1, self.max_disp // self.patch_size)
        device = left_feat.device
        dtype = left_feat.dtype
        cost = torch.zeros(B, C, max_disp_feat, H, W, device=device, dtype=dtype)
        for d in range(max_disp_feat):
            if d > 0:
                cost[:, :, d, :, d:] = left_feat[:, :, :, d:] - right_feat[:, :, :, :-d]
            else:
                cost[:, :, d, :, :] = left_feat - right_feat
        return cost

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor):
        h, w = left_image.shape[-2:]
        left_feat = self.get_features(left_image)
        right_feat = self.get_features(right_image)
        cost_volume = self.build_cost_volume(left_feat, right_feat)
        cost_aggregated = self.cost_aggregator(cost_volume).squeeze(1)
        cost_soft = F.softmax(-cost_aggregated, dim=1)
        max_disp_feat = cost_soft.shape[1]
        disp_values = torch.arange(0, max_disp_feat, device=cost_soft.device, dtype=torch.float32).view(1, -1, 1, 1)
        disp_feat = torch.sum(cost_soft * disp_values, 1, keepdim=True)
        disp = F.interpolate(disp_feat * self.patch_size, size=(h, w), mode='bilinear', align_corners=False)

        disp_right = None
        if self.cfg.USE_CONSISTENCY_LOSS:
            cost_vol_r = self.build_cost_volume(right_feat, left_feat)
            cost_agg_r = self.cost_aggregator(cost_vol_r).squeeze(1)
            cost_soft_r = F.softmax(-cost_agg_r, dim=1)
            disp_feat_r = torch.sum(cost_soft_r * disp_values, 1, keepdim=True)
            disp_right = F.interpolate(disp_feat_r * self.patch_size, size=(h, w), mode='bilinear', align_corners=False)

        return {"disparity": disp, "disparity_right": disp_right}

# -------------------------
# Loss & metrics
# -------------------------
class ImprovedSelfSupervisedLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ssim = SSIM()
        self.smoothness_weight = cfg.INITIAL_SMOOTHNESS_WEIGHT
        self.photometric_weights = cfg.PHOTOMETRIC_LOSS_WEIGHTS
        self.use_consistency = cfg.USE_CONSISTENCY_LOSS
        self.consistency_weight = cfg.CONSISTENCY_LOSS_WEIGHT

    def forward(self, inputs: dict, outputs: dict):
        left = inputs["left_image"]
        right = inputs["right_image"]
        mask = inputs.get("mask", torch.ones_like(left[:, :1, :, :], device=left.device))
        disp = outputs["disparity"]

        warped_right = self.inverse_warp(right, disp)
        warped_left = self.inverse_warp(left, -disp)

        mask_sum = mask.sum() + 1e-8

        l1_map_r = torch.abs(warped_right - left)
        l1_map_l = torch.abs(warped_left - right)
        l1_r = (l1_map_r * mask).sum() / mask_sum
        l1_l = (l1_map_l * mask).sum() / mask_sum
        l1_loss = 0.5 * (l1_r + l1_l)

        ssim_r = 1.0 - self.ssim(warped_right, left)
        ssim_l = 1.0 - self.ssim(warped_left, right)
        ssim_loss = 0.5 * (ssim_r + ssim_l)

        photometric_loss = self.photometric_weights[0] * ssim_loss + self.photometric_weights[1] * l1_loss

        smooth_loss = self.compute_smoothness_loss(disp, left)

        consistency_loss = torch.tensor(0.0, device=left.device)
        if self.use_consistency and ("disparity_right" in outputs) and (outputs["disparity_right"] is not None):
            disp_r = outputs["disparity_right"]
            warped_disp_r = self.inverse_warp(disp_r, -disp)
            consistency_loss = (torch.abs(disp - warped_disp_r) * mask).sum() / mask_sum

        total = photometric_loss + self.smoothness_weight * smooth_loss
        if self.use_consistency:
            total = total + self.consistency_weight * consistency_loss

        return {
            "total_loss": total,
            "photometric_loss": photometric_loss,
            "l1_loss": l1_loss,
            "ssim_loss": ssim_loss,
            "smoothness_loss": smooth_loss,
            "consistency_loss": consistency_loss,
            "warped_right_image": warped_right,
            "warped_left_image": warped_left
        }

    def inverse_warp(self, features: torch.Tensor, disp: torch.Tensor):
        B, C, H, W = features.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=features.device),
                                            torch.arange(W, device=features.device), indexing='ij')
        x_coords = x_coords.float()
        y_coords = y_coords.float()
        coords = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        disp = disp.squeeze(1)
        transformed_x = coords[:, 0, :, :] - disp
        grid_x = 2.0 * transformed_x / (W - 1) - 1.0
        grid_y = 2.0 * coords[:, 1, :, :] / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def compute_smoothness_loss(self, disp: torch.Tensor, img: torch.Tensor):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
        grad_disp_x = grad_disp_x * torch.exp(-grad_img_x)
        grad_disp_y = grad_disp_y * torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()

class EvaluationMetrics:
    @staticmethod
    def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return (20.0 * torch.log10(torch.tensor(max_val) / (torch.sqrt(mse) + 1e-8))).item()

    @staticmethod
    def compute_rmse(img1: torch.Tensor, img2: torch.Tensor):
        return torch.sqrt(torch.mean((img1 - img2) ** 2)).item()

    @staticmethod
    def compute_ssim(img1: torch.Tensor, img2: torch.Tensor):
        ssim_module = SSIM()
        return ssim_module(img1, img2).item()

    @staticmethod
    def evaluate_reconstruction(inputs: dict, loss_components: dict, cfg: Config):
        left = inputs["left_image"]
        warped_right = loss_components["warped_right_image"]
        psnr = EvaluationMetrics.compute_psnr(left, warped_right, max_val=cfg.PSNR_MAX_VAL)
        rmse = EvaluationMetrics.compute_rmse(left, warped_right)
        ssim = EvaluationMetrics.compute_ssim(left, warped_right)
        return {"psnr": psnr, "rmse": rmse, "ssim": ssim}

# -------------------------
# Trainer
# -------------------------
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class Trainer:
    def __init__(self, cfg: Config, left_dir: str, right_dir: str):
        self.cfg = cfg
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        os.makedirs(cfg.TENSORBOARD_DIR, exist_ok=True)
        if cfg.VISUALIZE_TRAINING:
            os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        if self.device.type == 'cpu':
            logging.warning("No GPU detected. Mixed precision disabled.")
            cfg.USE_MIXED_PRECISION = False

        self.writer = None
        if SummaryWriter:
            try:
                self.writer = SummaryWriter(log_dir=os.path.join(cfg.TENSORBOARD_DIR, datetime.now().strftime('%Y%m%d-%H%M%S')))
            except Exception:
                self.writer = None

        # Model, optimizer, loss, scheduler
        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.loss_fn = ImprovedSelfSupervisedLoss(cfg).to(self.device)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, cfg.NUM_EPOCHS))
        self.evaluator = EvaluationMetrics()

        train_ds = RectifiedWaveStereoDataset(cfg, left_dir, right_dir, is_validation=False)
        val_ds = RectifiedWaveStereoDataset(cfg, left_dir, right_dir, is_validation=True)
        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=self.device.type=='cuda', drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=self.device.type=='cuda')

        self.scaler = amp.GradScaler(enabled=(cfg.USE_MIXED_PRECISION and self.device.type=='cuda'))
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
                    return
            except Exception:
                continue

    def train(self):
        logging.info("=== Start self-supervised training ===")
        best_val_loss = float('inf')
        for epoch in range(self.cfg.NUM_EPOCHS):
            self.current_smoothness_weight *= self.cfg.SMOOTHNESS_WEIGHT_DECAY
            self.loss_fn.smoothness_weight = self.current_smoothness_weight

            self.model.train()
            epoch_loss = 0.0
            epoch_metrics = {'psnr': 0.0, 'rmse': 0.0, 'ssim': 0.0}
            count = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS} [train]")
            for batch in pbar:
                if batch is None:
                    continue
                left, right, mask = [t.to(self.device) for t in batch]
                inputs = {"left_image": left, "right_image": right, "mask": mask}
                self.optimizer.zero_grad()
                with amp.autocast(device_type=self.device.type, enabled=(self.cfg.USE_MIXED_PRECISION and self.device.type=='cuda')):
                    outputs = self.model(left, right)
                    loss_dict = self.loss_fn(inputs, outputs)
                    loss = loss_dict["total_loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss at step {self.step}, skipping.")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.cfg.GRADIENT_CLIP_VAL)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                metrics = self.evaluator.evaluate_reconstruction(inputs, loss_dict, self.cfg)
                epoch_loss += loss.item()
                for k in epoch_metrics: epoch_metrics[k] += metrics[k]
                count += 1
                pbar.set_postfix({'loss': loss.item(), 'psnr': metrics['psnr'], 'lr': self.optimizer.param_groups[0]['lr']})

                if self.writer:
                    self.writer.add_scalar('Loss/train_step', loss.item(), self.step)
                    self.writer.add_scalar('Metrics/train_psnr_step', metrics['psnr'], self.step)

                if self.cfg.VISUALIZE_TRAINING and (self.step % self.cfg.VISUALIZE_INTERVAL == 0):
                    self.visualize(inputs, outputs, loss_dict, self.step, phase='train')

                self.step += 1

            if count > 0:
                avg_loss = epoch_loss / count
                avg_metrics = {k: v / count for k, v in epoch_metrics.items()}
                self.loss_history['train']['total'].append(avg_loss)
                self.metric_history['train']['psnr'].append(avg_metrics['psnr'])
                self.metric_history['train']['ssim'].append(avg_metrics['ssim'])

            val_loss, val_metrics = self.validate(epoch)
            logging.info(f"Epoch {epoch+1}/{self.cfg.NUM_EPOCHS} -> val_loss: {val_loss:.4f}, PSNR: {val_metrics.get('psnr',0):.2f}")

            try:
                self.scheduler.step()
            except Exception:
                pass

            if not math.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model_self_supervised.pth")
                try:
                    torch.save(self.model.state_dict(), save_path)
                    logging.info(f"Saved best model to {save_path}")
                except Exception as e:
                    logging.error(f"Failed to save model: {e}")

            if self.cfg.VISUALIZE_TRAINING:
                self.plot_training_history()
            self.update_log_file(epoch)

        logging.info("Training finished.")
        if self.writer:
            self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'psnr': 0.0, 'ssim': 0.0, 'rmse': 0.0}
        cnt = 0
        last_loss_dict = None
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc='[val]')):
                if batch is None:
                    continue
                left, right, mask = [t.to(self.device) for t in batch]
                inputs = {"left_image": left, "right_image": right, "mask": mask}
                outputs = self.model(left, right)
                loss_dict = self.loss_fn(inputs, outputs)
                last_loss_dict = loss_dict
                if not torch.isnan(loss_dict["total_loss"]):
                    total_loss += loss_dict["total_loss"].item()
                metrics = self.evaluator.evaluate_reconstruction(inputs, loss_dict, self.cfg)
                for k in total_metrics: total_metrics[k] += metrics[k]
                cnt += 1
                if i == 0 and self.cfg.VISUALIZE_TRAINING:
                    self.visualize(inputs, outputs, loss_dict, epoch, phase='val')

        avg_loss = total_loss / cnt if cnt > 0 else float('nan')
        avg_metrics = {k: total_metrics[k] / cnt for k in total_metrics} if cnt > 0 else {}
        if cnt > 0:
            self.loss_history['val']['total'].append(avg_loss)
            self.metric_history['val']['psnr'].append(avg_metrics.get('psnr', 0))
            self.metric_history['val']['ssim'].append(avg_metrics.get('ssim', 0))
        return avg_loss, avg_metrics

    def visualize(self, inputs, outputs, loss_components, step, phase='train'):
        left_np = inputs["left_image"][0].detach().permute(1,2,0).cpu().numpy()
        warped_np = loss_components["warped_right_image"][0].detach().permute(1,2,0).cpu().numpy()
        disp_np = outputs["disparity"][0,0].detach().cpu().numpy()
        mask_np = inputs["mask"][0,0].detach().cpu().numpy()

        fig = plt.figure(figsize=(12,8))
        plt.suptitle(f"Visualization step {step} ({phase})")
        ax1 = plt.subplot(2,3,1); ax1.imshow(left_np); ax1.set_title("left"); ax1.axis('off')
        ax2 = plt.subplot(2,3,2); ax2.imshow(warped_np); ax2.set_title("warped_right"); ax2.axis('off')
        ax3 = plt.subplot(2,3,3); im = plt.imshow(disp_np, cmap='viridis'); plt.title("pred_disp"); plt.axis('off'); plt.colorbar(im, ax=ax3, fraction=0.046)
        ax4 = plt.subplot(2,3,4); diff = np.clip(np.abs(left_np - warped_np), 0, 1); ax4.imshow(diff.mean(axis=2), cmap='hot'); ax4.set_title("photometric error"); ax4.axis('off')
        ax5 = plt.subplot(2,3,5); ax5.imshow(mask_np, cmap='gray'); ax5.set_title("mask"); ax5.axis('off')
        metrics = EvaluationMetrics.evaluate_reconstruction(inputs, loss_components, self.cfg)
        ax6 = plt.subplot(2,3,6); ax6.axis('off'); ax6.text(0,0.5, f"total:{loss_components['total_loss'].item():.4f}\npsnr:{metrics['psnr']:.2f}\nssim:{metrics['ssim']:.4f}", fontsize=12)
        plt.tight_layout(rect=[0,0,1,0.96])

        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, f"{phase}_step_{step:06d}.png")
        try:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
        except Exception:
            plt.savefig(save_path)
        plt.close(fig)

        if self.writer:
            try:
                self.writer.add_image(f'vis/{phase}/left', torch.from_numpy(left_np).permute(2,0,1), global_step=step)
            except Exception:
                pass

    def plot_training_history(self):
        if not self.loss_history['train']['total']:
            return
        fig, axes = plt.subplots(2,2, figsize=(14,10))
        axes[0,0].plot(self.loss_history['train']['total'], label='train')
        if self.loss_history['val']['total']: axes[0,0].plot(self.loss_history['val']['total'], label='val')
        axes[0,0].set_title('Total Loss'); axes[0,0].legend(); axes[0,0].grid(True)

        axes[0,1].plot(self.loss_history['train'].get('photometric', []), label='photometric')
        axes[0,1].plot(self.loss_history['train'].get('smoothness', []), label='smoothness')
        axes[0,1].set_title('Loss Components'); axes[0,1].legend(); axes[0,1].grid(True)

        if self.metric_history['train']['psnr']: axes[1,0].plot(self.metric_history['train']['psnr'], label='train psnr')
        if self.metric_history['val'].get('psnr'): axes[1,0].plot(self.metric_history['val'].get('psnr'), label='val psnr')
        axes[1,0].set_title('PSNR'); axes[1,0].legend(); axes[1,0].grid(True)

        if self.metric_history['train']['ssim']: axes[1,1].plot(self.metric_history['train']['ssim'], label='train ssim')
        if self.metric_history['val'].get('ssim'): axes[1,1].plot(self.metric_history['val'].get('ssim'), label='val ssim')
        axes[1,1].set_title('SSIM'); axes[1,1].legend(); axes[1,1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, "training_history.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def update_log_file(self, epoch):
        log_data = {'config': asdict(self.cfg), 'epoch': epoch, 'loss_history': self.loss_history, 'metric_history': self.metric_history, 'update_time': datetime.now().isoformat()}
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Failed to write log file: {e}")

# -------------------------
# Entrypoint (main)
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="DINOv3 self-supervised trainer (fixed main)")
    parser.add_argument("--left_dir", type=str, default=None, help="Override left images directory")
    parser.add_argument("--right_dir", type=str, default=None, help="Override right images directory")
    parser.add_argument("--calib", type=str, default=None, help="Override calibration .npz file")
    parser.add_argument("--no_cuda", action='store_true', help="Force CPU")
    args = parser.parse_args()

    cfg = Config()
    if args.calib:
        cfg.CALIBRATION_FILE = args.calib
    if args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    left_candidate = args.left_dir if args.left_dir else cfg.LEFT_IMAGE_DIR
    right_candidate = args.right_dir if args.right_dir else cfg.RIGHT_IMAGE_DIR
    left_dir = resolve_image_dir(left_candidate, "left", cfg.PROJECT_ROOT)
    right_dir = resolve_image_dir(right_candidate, "right", cfg.PROJECT_ROOT)

    logging.info(f"Final LEFT_IMAGE_DIR = {left_dir}")
    logging.info(f"Final RIGHT_IMAGE_DIR = {right_dir}")
    logging.info(f"Calibration file = {cfg.CALIBRATION_FILE}")
    logging.info(f"DINO local path = {cfg.DINO_LOCAL_PATH}")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.TENSORBOARD_DIR, exist_ok=True)

    plt.switch_backend('Agg')

    trainer = Trainer(cfg, left_dir, right_dir)
    trainer.train()

if __name__ == "__main__":
    if SummaryWriter is None:
        logging.warning("TensorBoard not installed: SummaryWriter disabled (optional).")
    main()
