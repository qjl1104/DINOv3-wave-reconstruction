# optimized_dinov3_stereo_reconstruction.py
# DINO-Iterative-StereoNet: 融合DINO特征与迭代优化架构的自监督立体重建系统
# 主要设计: DINO特征主干, 可训练特征适配器, 迭代优化器

import os
import sys
import glob
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import exp, sqrt
import random

# --- 导入依赖库 ---
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("[WARNING]: TensorBoard unavailable")
    SummaryWriter = None
try:
    from transformers import AutoModel
except ImportError:
    print("[FATAL ERROR]: transformers library required")
    sys.exit(1)


# --- 全局辅助函数 ---
def safe_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# --- 配置类 ---
@dataclass
class DINO_Iterative_Config:
    """DINO-Iterative-StereoNet的配置参数"""
    # 路径
    PROJECT_ROOT: str = r"D:\Research\wave_reconstruction_project\DINOv3"
    DATA_ROOT: str = os.path.dirname(PROJECT_ROOT)
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")
    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs")
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")

    # 数据处理
    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 512
    MASK_THRESHOLD: int = 30

    # 核心模型参数
    ITERATIONS: int = 12
    FEATURE_CHANNELS: int = 768  # DINO-base的特征维度
    CONTEXT_CHANNELS: int = 128
    DINO_PATCH_SIZE: int = 16  # 修正: 根据错误日志推断patch size为16

    # 训练策略
    BATCH_SIZE: int = 2
    LEARNING_RATE: float = 2e-4
    NUM_EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.15
    GRADIENT_CLIP_VAL: float = 1.0
    GRADIENT_ACCUMULATION_STEPS: int = 2
    USE_MIXED_PRECISION: bool = True
    EARLY_STOPPING_PATIENCE: int = 15
    WARMUP_EPOCHS: int = 10

    # 损失函数
    LOSS_GAMMA: float = 0.85
    SMOOTHNESS_WEIGHT: float = 0.3

    # 数据增强
    USE_ADVANCED_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.8

    # 可视化
    VISUALIZE_TRAINING: bool = True


# --- 基础模块 ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm='instance', activation='relu'):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)]
        if norm == 'instance': layers.append(nn.InstanceNorm2d(out_ch))
        if activation == 'relu': layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# --- DINO-Iterative-StereoNet 核心架构 ---

class ContextEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32, stride=2), ConvBlock(32, 32),
            ConvBlock(32, 64, stride=2), ConvBlock(64, 64),
            ConvBlock(64, 96, stride=2), ConvBlock(96, 96),
            ConvBlock(96, out_channels)
        )

    def forward(self, image): return self.net(image)


class FeatureAdapter(nn.Module):
    """可训练的特征适配器 (即特征致密化网络)"""

    def __init__(self, in_channels, mid_channels=128):
        super().__init__()
        self.densify_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        )

    def forward(self, x): return x + self.densify_block(x)


class Correlation:
    """构建4D相关性体积"""

    def __init__(self, fmap1, fmap2, radius=4, stride=1):
        self.fmap1, self.fmap2 = fmap1, fmap2
        self.radius, self.stride = radius, stride
        self.corr_pyramid = self._build_pyramid()

    def _build_pyramid(self):
        B, C, H, W = self.fmap1.shape
        pyramid = []
        padded_fmap2 = F.pad(self.fmap2, [self.radius] * 4)
        for i in range(2 * self.radius // self.stride + 1):
            for j in range(2 * self.radius // self.stride + 1):
                shift_y, shift_x = i * self.stride, j * self.stride
                shifted = padded_fmap2[:, :, shift_y:shift_y + H, shift_x:shift_x + W]
                corr = torch.mean(self.fmap1 * shifted, dim=1)
                pyramid.append(corr)
        return torch.stack(pyramid, dim=1)

    def __call__(self, coords):
        B, _, H, W = coords.shape
        x_coords, y_coords = coords[:, 0], coords[:, 1]

        x_coords = x_coords / self.stride
        y_coords = y_coords / self.stride

        r = self.radius // self.stride

        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy, dx = torch.meshgrid(dy, dx, indexing='ij')

        grid_x = x_coords[:, None] + dx[None, :, :, None, None]
        grid_y = y_coords[:, None] + dy[None, :, :, None, None]

        grid = torch.stack([grid_x, grid_y], dim=-1).view(B, (2 * r + 1) ** 2, -1, 2)
        grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1

        return F.grid_sample(self.corr_pyramid.view(B, 1, -1, H * W), grid, align_corners=True,
                             padding_mode='border').view(B, -1, H, W)


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=128, corr_dim=81, motion_dim=1):
        super().__init__()
        self.gru = nn.Conv2d(hidden_dim + context_dim + corr_dim + motion_dim, hidden_dim * 3, 3, padding=1)
        self.disp_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, hidden_state, context, corr, disp):
        inp = torch.cat([hidden_state, context, corr, disp], dim=1)
        gate_z, gate_r, candidate_q_inp = torch.split(self.gru(inp), hidden_state.shape[1], dim=1)
        gate_z, gate_r = torch.sigmoid(gate_z), torch.sigmoid(gate_r)

        reset_hidden = gate_r * hidden_state
        candidate_q = torch.tanh(candidate_q_inp)

        new_hidden = (1 - gate_z) * hidden_state + gate_z * candidate_q
        delta_disp = self.disp_head(new_hidden)
        return new_hidden, delta_disp


class ConvexUpsampler(nn.Module):
    def __init__(self, in_channels, upsample_factor=14):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, upsample_factor ** 2 * 9, 1)
        )

    def forward(self, disp, motion_features):
        B, _, H, W = disp.shape
        mask = self.mask_predictor(motion_features).view(B, 1, 9, self.upsample_factor, self.upsample_factor, H, W)
        mask = torch.softmax(mask, dim=2)
        up_disp = F.unfold(disp * self.upsample_factor, [3, 3], padding=1).view(B, 1, 9, 1, 1, H, W)
        up_disp = torch.sum(mask * up_disp, dim=2)
        return up_disp.permute(0, 1, 4, 2, 5, 3).reshape(B, 1, self.upsample_factor * H, self.upsample_factor * W)


class DINO_Iterative_StereoNet(nn.Module):
    def __init__(self, cfg: DINO_Iterative_Config):
        super().__init__()
        self.cfg = cfg
        self.dino = self._load_dino()
        for param in self.dino.parameters(): param.requires_grad = False

        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

        self.feature_adapter = FeatureAdapter(cfg.FEATURE_CHANNELS)
        self.c_encoder = ContextEncoder(cfg.CONTEXT_CHANNELS)

        self.update_block = UpdateBlock(hidden_dim=cfg.CONTEXT_CHANNELS)
        self.upsampler = ConvexUpsampler(cfg.CONTEXT_CHANNELS, upsample_factor=cfg.DINO_PATCH_SIZE)

    def _load_dino(self):
        try:
            model = AutoModel.from_pretrained(self.cfg.DINO_LOCAL_PATH, local_files_only=True)
            print(f"✓ 成功加载DINOv3模型: {self.cfg.DINO_LOCAL_PATH}")
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")

    def get_dino_features(self, image):
        B, C, H, W = image.shape
        normalized_image = (image / 255.0 - torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1,
                                                                                                          1)) / torch.tensor(
            [0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)

        with torch.no_grad():
            features = self.dino(normalized_image, output_hidden_states=False).last_hidden_state

        start_index = 1 + self.num_register_tokens
        patch_tokens = features[:, start_index:, :]

        return patch_tokens.permute(0, 2, 1).reshape(B, self.cfg.FEATURE_CHANNELS, H // self.cfg.DINO_PATCH_SIZE,
                                                     W // self.cfg.DINO_PATCH_SIZE)

    def forward(self, left_image, right_image):
        fmap1 = self.get_dino_features(left_image)
        fmap2 = self.get_dino_features(right_image)

        fmap1 = self.feature_adapter(fmap1)
        fmap2 = self.feature_adapter(fmap2)

        context = self.c_encoder(left_image / 255.0)

        corr_fn = Correlation(fmap1, fmap2)

        B, _, H, W = fmap1.shape

        # 修正: 创建基础坐标网格
        coords_y, coords_x = torch.meshgrid(torch.arange(H, device=left_image.device),
                                            torch.arange(W, device=left_image.device), indexing='ij')
        coords0 = torch.stack([coords_x, coords_y], dim=0).float()
        coords0 = coords0.unsqueeze(0).repeat(B, 1, 1, 1)

        disp = torch.zeros(B, 1, H, W, device=left_image.device)
        hidden_state = torch.zeros(B, self.cfg.CONTEXT_CHANNELS, H, W, device=left_image.device)

        disp_predictions = []
        for _ in range(self.cfg.ITERATIONS):
            disp = disp.detach()

            # 修正: 从1D视差创建2D流场，并添加到基础网格
            flow = torch.cat([disp, torch.zeros_like(disp)], dim=1)
            current_coords = coords0 + flow

            corr = corr_fn(current_coords)
            hidden_state, delta_disp = self.update_block(hidden_state, context, corr, disp)
            disp = disp + delta_disp

            disp_up = self.upsampler(disp, hidden_state)
            disp_predictions.append(disp_up)

        return disp_predictions


# --- 损失函数, 数据集, 训练器 ---

class EnhancedSSIM(nn.Module):
    """SSIM损失函数"""

    def __init__(self, window_size=11, channel=3):
        super(EnhancedSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return torch.clamp((1 - ssim_map) / 2, 0, 1)


class IterativeLoss(nn.Module):
    def __init__(self, cfg: DINO_Iterative_Config):
        super().__init__()
        self.cfg = cfg
        self.ssim = EnhancedSSIM()

    def compute_photometric_loss(self, img1, img2, disp, mask):
        img1, img2 = img1 / 255.0, img2 / 255.0  # Normalize for loss
        warped_img2 = self.inverse_warp(img2, disp)
        l1_loss = (torch.abs(warped_img2 - img1) * mask).sum() / (mask.sum() + 1e-8)
        ssim_loss = (self.ssim(warped_img2, img1) * mask).sum() / (mask.sum() + 1e-8)
        return 0.85 * ssim_loss + 0.15 * l1_loss, warped_img2

    def compute_smoothness_loss(self, disp, img, mask):
        img = img / 255.0
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        smoothness_x = (grad_disp_x * torch.exp(-grad_img_x) * mask[:, :, :, :-1]).mean()
        smoothness_y = (grad_disp_y * torch.exp(-grad_img_y) * mask[:, :, :-1, :]).mean()
        return smoothness_x + smoothness_y

    def inverse_warp(self, features, disp):
        B, C, H, W = features.shape
        device = features.device
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                            torch.arange(W, device=device, dtype=torch.float32), indexing='ij')

        pixel_coords = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] - disp.squeeze(1)

        pixel_coords[:, 0, :, :] = 2 * pixel_coords[:, 0, :, :] / (W - 1) - 1
        pixel_coords[:, 1, :, :] = 2 * pixel_coords[:, 1, :, :] / (H - 1) - 1

        return F.grid_sample(features, pixel_coords.permute(0, 2, 3, 1), align_corners=True, padding_mode='border')

    def forward(self, inputs, outputs):
        left, right, mask = inputs['left_image'], inputs['right_image'], inputs['mask']
        total_loss = 0

        for i, pred_disp in enumerate(outputs):
            gamma = self.cfg.LOSS_GAMMA ** (self.cfg.ITERATIONS - i - 1)
            photometric_loss, warped = self.compute_photometric_loss(left, right, pred_disp, mask)
            smoothness_loss = self.compute_smoothness_loss(pred_disp, left, mask)
            total_loss += gamma * (photometric_loss + self.cfg.SMOOTHNESS_WEIGHT * smoothness_loss)

        return {'total_loss': total_loss, 'warped_right_image': warped}


class StereoDataset(Dataset):
    def __init__(self, cfg: DINO_Iterative_Config, is_validation=False):
        self.cfg, self.is_validation = cfg, is_validation
        self.augmenter = AdvancedDataAugmentation(cfg) if not is_validation and cfg.USE_ADVANCED_AUGMENTATION else None
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        calib = np.load(cfg.CALIBRATION_FILE)
        self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
        self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
        self.roi_left, self.roi_right = tuple(calib['roi_left']), tuple(calib['roi_right'])

        indices = np.arange(len(self.left_images))
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(len(self.left_images) * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]
        print(f"✓ 数据集: {'验证' if is_validation else '训练'}集 {len(self.indices)} 样本")

    def _load_frame(self, idx):
        left_path = self.left_images[idx]
        right_path = left_path.replace("left_images", "right_images").replace("left", "right")
        if not os.path.exists(right_path): return None
        left_raw, right_raw = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE), cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if left_raw is None or right_raw is None: return None

        left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

        x, y, w, h = self.roi_left
        left_rect = left_rect[y:y + h, x:x + w]
        x, y, w, h = self.roi_right
        right_rect = right_rect[y:y + h, x:x + w]

        left_img = cv2.resize(left_rect, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
        right_img = cv2.resize(right_rect, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
        _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        return left_img, right_img, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        try:
            result = self._load_frame(self.indices[i])
            if result is None: return None
            left, right, mask = result
            if self.augmenter: left, right = self.augmenter.apply_individual_augmentations(left, right)

            to_tensor = lambda img: torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float()
            return to_tensor(left), to_tensor(right), torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        except Exception:
            return None


class AdvancedDataAugmentation:  # Simplified version
    def __init__(self, cfg: DINO_Iterative_Config):
        self.cfg = cfg

    def apply_individual_augmentations(self, left, right):
        if np.random.rand() < 0.5:  # Asymmetric color aug
            left_aug = np.copy(left)
            if np.random.rand() < 0.8: factor = np.random.uniform(0.7, 1.4); left_aug = np.clip(left_aug * factor, 0,
                                                                                                255)
            if np.random.rand() < 0.4: factor = np.random.uniform(0.7, 1.4); mean = left_aug.mean(); left_aug = np.clip(
                (left_aug - mean) * factor + mean, 0, 255)
            left = left_aug.astype(np.uint8)
        return left, right


class Trainer:
    def __init__(self, cfg: DINO_Iterative_Config):
        self.cfg, self.device = cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(self.cfg.RUNS_BASE_DIR, f"DINO-Iterative_{self.timestamp}")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = DINO_Iterative_StereoNet(cfg).to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS,
                                                              eta_min=1e-6)
        self.loss_fn = IterativeLoss(cfg).to(self.device)
        self.scaler = torch.amp.GradScaler(enabled=(cfg.USE_MIXED_PRECISION and self.device.type == 'cuda'))
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tensorboard")) if SummaryWriter else None

        train_dataset, val_dataset = StereoDataset(cfg), StereoDataset(cfg, is_validation=True)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2,
                                       pin_memory=True, collate_fn=safe_collate_fn, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2,
                                     pin_memory=True, collate_fn=safe_collate_fn, drop_last=True)

        self.best_val_loss, self.epochs_no_improve = float('inf'), 0
        print("✓ 组件初始化完成")

    def train(self):
        for epoch in range(self.cfg.NUM_EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} ---")
            if epoch < self.cfg.WARMUP_EPOCHS:
                for pg in self.optimizer.param_groups: pg['lr'] = self.cfg.LEARNING_RATE * (
                            epoch + 1) / self.cfg.WARMUP_EPOCHS

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch(epoch)

            if epoch >= self.cfg.WARMUP_EPOCHS - 1: self.scheduler.step()
            print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            if self.writer: self.writer.add_scalars('Loss/Epoch', {'Train': train_loss, 'Val': val_loss}, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss, self.epochs_no_improve = val_loss, 0
                torch.save({'model_state_dict': self.model.state_dict(), 'config': asdict(self.cfg)},
                           os.path.join(self.checkpoint_dir, 'best_model.pth'))
                print(f"✓ 最佳模型已保存 (损失: {self.best_val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.cfg.EARLY_STOPPING_PATIENCE: print("早停触发"); break

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")
        self.optimizer.zero_grad()
        for i, batch in enumerate(pbar):
            if batch is None: continue
            left, right, mask = [d.to(self.device) for d in batch]
            with torch.amp.autocast(device_type=self.device.type, enabled=self.cfg.USE_MIXED_PRECISION):
                preds = self.model(left, right)
                loss_dict = self.loss_fn({'left_image': left, 'right_image': right, 'mask': mask}, preds)
                loss = loss_dict['total_loss'] / self.cfg.GRADIENT_ACCUMULATION_STEPS
            self.scaler.scale(loss).backward()
            if (i + 1) % self.cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            total_loss += loss_dict['total_loss'].item()
            pbar.set_postfix({'Loss': f"{loss_dict['total_loss'].item():.4f}"})
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        for i, batch in enumerate(self.val_loader):
            if batch is None: continue
            left, right, mask = [d.to(self.device) for d in batch]
            with torch.amp.autocast(device_type=self.device.type, enabled=self.cfg.USE_MIXED_PRECISION):
                preds = self.model(left, right)
                loss_dict = self.loss_fn({'left_image': left, 'right_image': right, 'mask': mask}, preds)
            total_loss += loss_dict['total_loss'].item()
            if i == 0 and self.cfg.VISUALIZE_TRAINING: self.visualize(left, preds, mask, epoch)
        return total_loss / len(self.val_loader)

    def visualize(self, left, preds, mask, epoch):
        left_img = left[0].cpu().permute(1, 2, 0).byte().numpy()
        pred_disp = preds[-1][0, 0].cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(left_img)
        axes[0].set_title('Input Image')
        im = axes[1].imshow(pred_disp, cmap='jet')
        axes[1].set_title('Predicted Disparity')
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f'val_epoch_{epoch + 1}.png'))
        plt.close()


def main():
    cfg = DINO_Iterative_Config()
    plt.switch_backend('Agg')
    trainer = Trainer(cfg)
    with open(os.path.join(trainer.run_dir, 'config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    trainer.train()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    main()

