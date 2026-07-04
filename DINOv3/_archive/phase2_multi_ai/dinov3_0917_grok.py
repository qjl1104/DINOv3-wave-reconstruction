
# dinov3_self_supervised_full_optimized_no_unfreeze.py
# 修复版：在 v914bfcde 和 b69e7f66 基础上修复以下问题：
# 1. CompletionNet SE 模块通道数不匹配问题（256通道输入改为128通道）
# 2. ImprovedSelfSupervisedLoss.forward 中处理 completion_pred 为 None 的情况
# 3. Trainer.train 和 Trainer.validate 中处理 loss_history 更新时的 KeyError
# 4. Trainer.validate 中修复 NameError: epoch_train_metrics 改为 epoch_val_metrics
# 5. 保留之前修复：Trainer.visualize 中添加 .detach() 解决 numpy() 错误
# 6. 保留之前修复：RectifiedWaveStereoDataset.__getitem__ 中 noise 未定义问题
# 原优化保留：
# - 不解冻DINOv3权重（requires_grad=False）
# - 增强TIL模块（2层GRU，hidden_size=64，残差连接）
# - 改进CompletionNet（加深U-Net，SE注意力，Dropout 0.3）
# - 优化FusionNet（多层Conv，BatchNorm，Dropout 0.3）
# - 调整损失权重（TEMPORAL_LOSS_WEIGHT=0.3, DISP_COMPLETION_WEIGHT=0.2）
# - 学习率温启（5 epoch从1e-5到1e-4）
# - Early Stopping（10 epoch无改进）
# - 数据增强（翻转、噪声，同步下一帧）
# - 数据加载效率（num_workers=4，pin_memory）
# - 时序一致性度量和差值可视化

import os
import sys
import glob
from dataclasses import dataclass
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

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

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
    PHOTOMETRIC_LOSS_WEIGHTS: tuple = (0.7, 0.3)
    USE_CONSISTENCY_LOSS: bool = True
    CONSISTENCY_LOSS_WEIGHT: float = 0.1

    INITIAL_SMOOTHNESS_WEIGHT: float = 0.5
    SMOOTHNESS_WEIGHT_DECAY: float = 0.98

    USE_TEMPORAL_LOSS: bool = True
    TEMPORAL_LOSS_WEIGHT: float = 0.3
    USE_TIL: bool = True
    USE_DISP_COMPLETION: bool = True
    USE_GRAD_REFINE: bool = True
    USE_FUSION_NET: bool = True
    USE_DYNAMIC_MASK: bool = True
    GRAD_REFINE_WEIGHT: float = 0.1
    DISP_COMPLETION_WEIGHT: float = 0.2
    MOTION_THRESH: float = 0.1


# --- SSIM ---
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None

    def gaussian(self, window_size, sigma):
        vals = [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
        gauss = torch.tensor(vals, dtype=torch.float32)
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D @ _1D.t()
        w = _2D.unsqueeze(0).unsqueeze(0)
        w = w.expand(channel, 1, window_size, window_size).contiguous()
        return w

    def forward(self, x, y):
        _, channel, _, _ = x.size()
        if self.window is None or self.channel != channel or self.window.device != x.device or self.window.dtype != x.dtype:
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


# --- CompletionNet ---
class CompletionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up3 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256 // 16, 1),  # 修复：匹配 down3 的 256 通道
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 16, 256, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        se_weight = self.se(d3)
        d3 = d3 * se_weight
        d3 = self.dropout(d3)
        u1 = self.relu(self.up1(d3))
        u1 = F.interpolate(u1, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.relu(self.up2(u1 + d2))
        u2 = self.dropout(u2)
        u2 = F.interpolate(u2, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        out = self.up3(u2 + d1)
        return out


# --- TILDecoder ---
class TILDecoder(nn.Module):
    def __init__(self, hidden_size=64, output_height=256, output_width=512):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, 64, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 1, 3, padding=1)
        self.residual_conv = nn.Conv2d(1, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, x, input_disp=None):
        B, hidden_size = x.shape
        sqrt_dim = int(math.sqrt(hidden_size)) if math.sqrt(hidden_size).is_integer() else 8
        x = x.view(B, hidden_size, 1, 1)
        x = F.interpolate(x, size=(sqrt_dim, sqrt_dim), mode='bilinear', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.up1(x))
        x = F.interpolate(x, size=(self.output_height, self.output_width), mode='bilinear', align_corners=False)
        out = self.conv2(x)
        if input_disp is not None:
            residual = self.residual_conv(input_disp)
            out = out + residual
        return out


# --- FusionNet ---
class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, curr, prev):
        x = torch.cat([curr, prev], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return self.conv3(x)


# --- ImprovedSelfSupervisedLoss ---
class ImprovedSelfSupervisedLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.smoothness_weight = cfg.INITIAL_SMOOTHNESS_WEIGHT
        self.photometric_weights = cfg.PHOTOMETRIC_LOSS_WEIGHTS
        self.use_consistency_loss = cfg.USE_CONSISTENCY_LOSS
        self.consistency_weight = cfg.CONSISTENCY_LOSS_WEIGHT
        self.use_temporal_loss = cfg.USE_TEMPORAL_LOSS
        self.temporal_weight = cfg.TEMPORAL_LOSS_WEIGHT
        self.use_grad_refine = cfg.USE_GRAD_REFINE
        self.grad_refine_weight = cfg.GRAD_REFINE_WEIGHT
        self.use_disp_completion = cfg.USE_DISP_COMPLETION
        self.disp_completion_weight = cfg.DISP_COMPLETION_WEIGHT
        self.use_dynamic_mask = cfg.USE_DYNAMIC_MASK
        self.motion_thresh = cfg.MOTION_THRESH
        self.ssim = SSIM()

    def forward(self, inputs, outputs):
        left_img = inputs["left_image"]
        right_img = inputs["right_image"]
        mask = inputs.get("mask", torch.ones_like(left_img[:, :1, :, :], device=left_img.device))
        pred_disp = outputs["disparity"]

        # 初始化所有损失组件
        loss_components = {
            "total_loss": 0.0,
            "photometric_loss": 0.0,
            "smoothness_loss": 0.0,
            "consistency_loss": 0.0,
            "temporal_loss": 0.0,
            "grad_refine_loss": 0.0,
            "disp_completion_loss": 0.0,
            "warped_right_image": None,
            "warped_left_image": None
        }

        # 调试日志：检查 inputs 和 outputs
        logging.debug(f"Inputs keys: {list(inputs.keys())}")
        logging.debug(f"Outputs keys: {list(outputs.keys())}")

        if self.use_dynamic_mask and "left_next" in inputs:
            motion_diff = torch.mean(torch.abs(left_img - inputs["left_next"]), dim=1, keepdim=True)
            motion_mask = (motion_diff > self.motion_thresh).float()
            mask = mask * (1 - motion_mask * 0.5)

        mask_sum = mask.sum() + 1e-8
        if mask_sum < 1e-6:
            logging.warning("Mask sum is near zero, setting photometric_loss to 0.0")
            l1_loss = torch.tensor(0.0, device=left_img.device)
            ssim_loss = torch.tensor(0.0, device=left_img.device)
            photometric_loss = torch.tensor(0.0, device=left_img.device)
        else:
            warped_right_image = self.inverse_warp(right_img, pred_disp)
            warped_left_image = self.inverse_warp(left_img, -pred_disp)

            l1_loss_right_map = torch.abs(warped_right_image - left_img)
            l1_loss_right = (l1_loss_right_map * mask).sum() / mask_sum
            l1_loss_left_map = torch.abs(warped_left_image - right_img)
            l1_loss_left = (l1_loss_left_map * mask).sum() / mask_sum
            l1_loss = (l1_loss_right + l1_loss_left) / 2

            ssim_loss_right_map = 1.0 - self.ssim(warped_right_image, left_img)
            ssim_loss_right = (ssim_loss_right_map * mask).sum() / mask_sum
            ssim_loss_left_map = 1.0 - self.ssim(warped_left_image, right_img)
            ssim_loss_left = (ssim_loss_left_map * mask).sum() / mask_sum
            ssim_loss = (ssim_loss_right + ssim_loss_left) / 2

            photometric_loss = self.photometric_weights[0] * ssim_loss + self.photometric_weights[1] * l1_loss
            loss_components["warped_right_image"] = warped_right_image
            loss_components["warped_left_image"] = warped_left_image
            loss_components["photometric_loss"] = photometric_loss

        smoothness_loss = self.compute_smoothness_loss(pred_disp, left_img)
        loss_components["smoothness_loss"] = smoothness_loss

        grad_refine_loss = 0.0
        if self.use_grad_refine:
            grad_disp_x = torch.abs(pred_disp[:, :, :, :-1] - pred_disp[:, :, :, 1:])
            grad_disp_y = torch.abs(pred_disp[:, :, :-1, :] - pred_disp[:, :, 1:, :])
            target_grad_x = torch.mean(torch.abs(left_img[:, :, :, :-1] - left_img[:, :, :, 1:]), 1, keepdim=True)
            target_grad_y = torch.mean(torch.abs(left_img[:, :, :-1, :] - left_img[:, :, 1:, :]), 1, keepdim=True)
            grad_refine_loss = F.l1_loss(grad_disp_x, target_grad_x) + F.l1_loss(grad_disp_y, target_grad_y)
            loss_components["grad_refine_loss"] = grad_refine_loss

        consistency_loss = 0.0
        if self.use_consistency_loss and "disparity_right" in outputs and outputs["disparity_right"] is not None:
            disp_left = outputs["disparity"]
            disp_right = outputs["disparity_right"]
            warped_disp_right = self.inverse_warp(disp_right, -disp_left)
            consistency_loss = (torch.abs(disp_left - warped_disp_right) * mask).sum() / mask_sum
            loss_components["consistency_loss"] = consistency_loss

        temporal_loss = 0.0
        disp_completion_loss = 0.0
        if self.use_temporal_loss and "disparity_next" in outputs and outputs["disparity_next"] is not None:
            pred_disp_next = outputs["disparity_next"]
            temporal_loss = (torch.abs(pred_disp - pred_disp_next) * mask).sum() / mask_sum
            loss_components["temporal_loss"] = temporal_loss

            if self.use_disp_completion and "init_disp" in outputs and outputs["init_disp"] is not None and outputs[
                "completion_pred"] is not None:
                completion_pred = outputs["completion_pred"]
                disp_completion_loss = F.l1_loss(pred_disp, completion_pred, reduction='mean')
                loss_components["disp_completion_loss"] = disp_completion_loss
            else:
                logging.debug("Skipping disp_completion_loss due to missing or invalid completion_pred")

        photometric_loss_next = 0.0
        smoothness_loss_next = 0.0
        consistency_loss_next = 0.0
        if self.use_temporal_loss and "left_next" in inputs and "disparity_next" in outputs and outputs[
            "disparity_next"] is not None:
            left_next = inputs["left_next"]
            right_next = inputs["right_next"]
            pred_disp_next = outputs["disparity_next"]
            mask_next = inputs.get("mask_next", torch.ones_like(left_next[:, :1, :, :], device=left_next.device))
            if self.use_dynamic_mask:
                motion_diff_next = torch.mean(torch.abs(left_next - left_img), dim=1, keepdim=True)
                motion_mask_next = (motion_diff_next > self.motion_thresh).float()
                mask_next = mask_next * (1 - motion_mask_next * 0.5)
            mask_next_sum = mask_next.sum() + 1e-8

            if mask_next_sum < 1e-6:
                logging.warning("Mask_next sum is near zero, setting photometric_loss_next to 0.0")
                l1_loss_next = torch.tensor(0.0, device=left_img.device)
                ssim_loss_next = torch.tensor(0.0, device=left_img.device)
                photometric_loss_next = torch.tensor(0.0, device=left_img.device)
            else:
                warped_right_next = self.inverse_warp(right_next, pred_disp_next)
                warped_left_next = self.inverse_warp(left_next, -pred_disp_next)

                l1_loss_right_next_map = torch.abs(warped_right_next - left_next)
                l1_loss_right_next = (l1_loss_right_next_map * mask_next).sum() / mask_next_sum
                l1_loss_left_next_map = torch.abs(warped_left_next - right_next)
                l1_loss_left_next = (l1_loss_left_next_map * mask_next).sum() / mask_next_sum
                l1_loss_next = (l1_loss_right_next + l1_loss_left_next) / 2

                ssim_loss_right_next_map = 1.0 - self.ssim(warped_right_next, left_next)
                ssim_loss_right_next = (ssim_loss_right_next_map * mask_next).sum() / mask_next_sum
                ssim_loss_left_next_map = 1.0 - self.ssim(warped_left_next, right_next)
                ssim_loss_left_next = (ssim_loss_left_next_map * mask_next).sum() / mask_next_sum
                ssim_loss_next = (ssim_loss_right_next + ssim_loss_left_next) / 2

                photometric_loss_next = self.photometric_weights[0] * ssim_loss_next + self.photometric_weights[
                    1] * l1_loss_next
                smoothness_loss_next = self.compute_smoothness_loss(pred_disp_next, left_next)

                if self.use_consistency_loss and "disparity_right_next" in outputs and outputs[
                    "disparity_right_next"] is not None:
                    disp_left_next = outputs["disparity_next"]
                    disp_right_next = outputs["disparity_right_next"]
                    warped_disp_right_next = self.inverse_warp(disp_right_next, -disp_left_next)
                    consistency_loss_next = (torch.abs(
                        disp_left_next - warped_disp_right_next) * mask_next).sum() / mask_next_sum

            photometric_loss = (photometric_loss + photometric_loss_next) / 2
            smoothness_loss = (smoothness_loss + smoothness_loss_next) / 2
            consistency_loss = (consistency_loss + consistency_loss_next) / 2
        else:
            logging.debug("Skipping temporal loss components due to missing disparity_next or left_next")

        total_loss = photometric_loss + self.smoothness_weight * smoothness_loss
        if self.use_consistency_loss:
            total_loss += self.consistency_weight * consistency_loss
        if self.use_temporal_loss and temporal_loss != 0.0:
            total_loss += self.temporal_weight * temporal_loss
        if self.use_grad_refine:
            total_loss += self.grad_refine_weight * grad_refine_loss
        if self.use_disp_completion and disp_completion_loss != 0.0:
            total_loss += self.disp_completion_weight * disp_completion_loss

        loss_components["total_loss"] = total_loss

        return loss_components

    def inverse_warp(self, features, disp):
        if features is None:
            raise ValueError("inverse_warp: features tensor is None")
        if features.dim() != 4:
            raise ValueError(
                f"inverse_warp: Expected 4D features tensor, got {features.dim()}D tensor with shape {features.shape}")

        B, C, H, W = features.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=features.device),
                                            torch.arange(W, device=features.device), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=0).float()
        pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)
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


# --- Dataset ---
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

        if cfg.USE_TEMPORAL_LOSS:
            self.indices = self.indices[:-1]

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

        _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

        left_next_tensor = None
        right_next_tensor = None
        mask_next_tensor = None
        if self.cfg.USE_TEMPORAL_LOSS:
            next_frame_idx = int(self.indices[idx + 1]) if idx + 1 < len(self.indices) else frame_idx
            left_next_path = self.left_images[next_frame_idx]
            next_basename = os.path.basename(left_next_path)
            if next_basename.startswith('left'):
                right_next_basename = 'right' + next_basename[4:]
            else:
                right_next_basename = next_basename
            right_next_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_next_basename)

            if os.path.exists(right_next_path):
                left_next_raw = cv2.imread(left_next_path, cv2.IMREAD_GRAYSCALE)
                right_next_raw = cv2.imread(right_next_path, cv2.IMREAD_GRAYSCALE)
                if left_next_raw is not None and right_next_raw is not None:
                    left_next_rect = cv2.remap(left_next_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
                    right_next_rect = cv2.remap(right_next_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
                    left_next_rect = left_next_rect[y:y + h, x:x + w]
                    right_next_rect = right_next_rect[y:y + h, x:x + w]
                    left_next = cv2.resize(left_next_rect, (target_w, target_h))
                    right_next = cv2.resize(right_next_rect, (target_w, target_h))
                    _, mask_next = cv2.threshold(left_next, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
                    mask_next_tensor = torch.from_numpy(mask_next).float().unsqueeze(0) / 255.0
                    left_next_bgr = cv2.cvtColor(left_next, cv2.COLOR_GRAY2BGR)
                    right_next_bgr = cv2.cvtColor(right_next, cv2.COLOR_GRAY2BGR)
                    left_next_tensor = torch.from_numpy(left_next_bgr.transpose(2, 0, 1)).float() / 255.0
                    right_next_tensor = torch.from_numpy(right_next_bgr.transpose(2, 0, 1)).float() / 255.0

        if not self.is_validation and self.cfg.USE_DATA_AUGMENTATION and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
            logging.debug(f"Applying data augmentation to sample {idx}")
            brightness_factor = 0.8 + 0.4 * np.random.rand()
            contrast_factor = 0.8 + 0.4 * np.random.rand()
            noise = np.random.normal(0, 10, left_img_bgr.shape).astype(np.uint8)

            left_img_bgr = np.clip(left_img_bgr * brightness_factor, 0, 255).astype(np.uint8)
            right_img_bgr = np.clip(right_img_bgr * brightness_factor, 0, 255).astype(np.uint8)

            left_mean, right_mean = left_img_bgr.mean(), right_img_bgr.mean()
            left_img_bgr = np.clip((left_img_bgr - left_mean) * contrast_factor + left_mean, 0, 255).astype(np.uint8)
            right_img_bgr = np.clip((right_img_bgr - right_mean) * contrast_factor + right_mean, 0, 255).astype(
                np.uint8)

            do_flip = np.random.rand() < 0.5
            if do_flip:
                left_img_bgr = cv2.flip(left_img_bgr, 1)
                right_img_bgr = cv2.flip(right_img_bgr, 1)

            if np.random.rand() < 0.3:
                left_img_bgr = np.clip(left_img_bgr + noise, 0, 255).astype(np.uint8)
                right_img_bgr = np.clip(right_img_bgr + noise, 0, 255).astype(np.uint8)

            if self.cfg.USE_TEMPORAL_LOSS and left_next_tensor is not None:
                left_next_bgr = (left_next_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                right_next_bgr = (right_next_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                left_next_bgr = np.clip(left_next_bgr * brightness_factor, 0, 255).astype(np.uint8)
                right_next_bgr = np.clip(right_next_bgr * brightness_factor, 0, 255).astype(np.uint8)
                left_next_mean, right_next_mean = left_next_bgr.mean(), right_next_bgr.mean()
                left_next_bgr = np.clip((left_next_bgr - left_next_mean) * contrast_factor + left_next_mean, 0,
                                        255).astype(np.uint8)
                right_next_bgr = np.clip((right_next_bgr - right_next_mean) * contrast_factor + right_next_mean, 0,
                                         255).astype(np.uint8)
                if do_flip:
                    left_next_bgr = cv2.flip(left_next_bgr, 1)
                    right_next_bgr = cv2.flip(right_next_bgr, 1)
                if np.random.rand() < 0.3:
                    left_next_bgr = np.clip(left_next_bgr + noise, 0, 255).astype(np.uint8)
                    right_next_bgr = np.clip(right_next_bgr + noise, 0, 255).astype(np.uint8)
                left_next_tensor = torch.from_numpy(left_next_bgr.transpose(2, 0, 1)).float() / 255.0
                right_next_tensor = torch.from_numpy(right_next_bgr.transpose(2, 0, 1)).float() / 255.0
                logging.debug(
                    f"Data augmentation applied: brightness={brightness_factor:.2f}, contrast={contrast_factor:.2f}, flip={do_flip}, noise=True")

        if self.cfg.USE_TEMPORAL_LOSS and left_next_tensor is None:
            left_next_tensor = left_tensor.clone()
            right_next_tensor = right_tensor.clone()
            mask_next_tensor = mask_tensor.clone()

        left_tensor = torch.from_numpy(left_img_bgr.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img_bgr.transpose(2, 0, 1)).float() / 255.0

        if self.cfg.USE_TEMPORAL_LOSS:
            return left_tensor, right_tensor, mask_tensor, left_next_tensor, right_next_tensor, mask_next_tensor
        else:
            return left_tensor, right_tensor, mask_tensor


# --- 3D hourglass and conv blocks ---
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


# --- DINOv3StereoModel ---
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

        self.feature_dim = getattr(self.dino.config, 'hidden_size', 768)
        self.patch_size = getattr(self.dino.config, 'patch_size', 8)
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

        self.cost_aggregator = nn.Sequential(
            conv_block_3d(self.feature_dim, 32),
            Hourglass3D(32),
            nn.Conv3d(32, 1, 3, padding=1)
        )

        if self.cfg.USE_TIL:
            self.til_gru = nn.GRU(1, 64, num_layers=2, batch_first=True, bidirectional=False)
            self.til_decoder = TILDecoder(hidden_size=64, output_height=cfg.IMAGE_HEIGHT, output_width=cfg.IMAGE_WIDTH)
        if self.cfg.USE_DISP_COMPLETION:
            self.completion_net = CompletionNet()
        if self.cfg.USE_FUSION_NET:
            self.fusion_net = FusionNet()

        logging.info("✓ 模型构建完成 (3D沙漏 + 优化模块)。")

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

    def compute_disparity(self, left_image, right_image, prev_disp=None):
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
        logging.debug(f"compute_disparity: input shape {left_image.shape}, output disparity shape {disparity.shape}")
        return disparity

    def forward(self, left_image, right_image, prev_disp=None, left_next=None, right_next=None):
        h, w = left_image.shape[-2:]
        disparity = self.compute_disparity(left_image, right_image, prev_disp)

        init_disp = None
        completion_pred = None
        if self.cfg.USE_DISP_COMPLETION and prev_disp is not None:
            if prev_disp.dim() == 3:
                prev_disp = prev_disp.unsqueeze(1)
            elif prev_disp.dim() != 4:
                logging.warning(f"Invalid prev_disp shape in disp completion: {prev_disp.shape}, skipping")
            else:
                try:
                    if prev_disp.size(0) != disparity.size(0):
                        logging.warning(
                            f"Batch size mismatch in disp completion: prev_disp {prev_disp.shape}, disparity {disparity.shape}, skipping")
                    else:
                        init_disp = self.inverse_warp(prev_disp, disparity)
                        completion_pred = self.completion_net(init_disp)
                        disparity = disparity + completion_pred
                except Exception as e:
                    logging.warning(f"Error in disp completion: {e}, skipping")

        if self.cfg.USE_TIL and prev_disp is not None:
            if prev_disp.dim() == 3:
                prev_disp = prev_disp.unsqueeze(1)
            if prev_disp.dim() == 4 and prev_disp.size(1) == 1:
                try:
                    if prev_disp.size(0) != disparity.size(0):
                        logging.warning(
                            f"Batch size mismatch in TIL: prev_disp {prev_disp.shape}, disparity {disparity.shape}, skipping")
                    else:
                        prev_disp = prev_disp.squeeze(1)
                        disparity_squeezed = disparity.squeeze(1)
                        disp_seq = torch.stack([prev_disp, disparity_squeezed], dim=1)
                        disp_seq = disp_seq.mean(dim=(2, 3), keepdim=True).squeeze(-1)
                        disp_seq = disp_seq.contiguous()
                        logging.debug(
                            f"TIL disp_seq shape: {disp_seq.shape}, is_contiguous: {disp_seq.is_contiguous()}")
                        _, til_hidden = self.til_gru(disp_seq)
                        til_hidden = til_hidden[-1]
                        til_disp = self.til_decoder(til_hidden, disparity)
                        disparity = disparity + til_disp
                except Exception as e:
                    logging.warning(f"Error in TIL module: {e}, skipping")
            else:
                logging.warning(f"Invalid prev_disp shape in TIL: {prev_disp.shape}, skipping")

        if self.cfg.USE_FUSION_NET and prev_disp is not None:
            if prev_disp.dim() == 3:
                prev_disp = prev_disp.unsqueeze(1)
            if prev_disp.dim() == 4:
                try:
                    if prev_disp.size(0) != disparity.size(0):
                        logging.warning(
                            f"Batch size mismatch in fusion net: prev_disp {prev_disp.shape}, disparity {disparity.shape}, skipping")
                    else:
                        disparity = self.fusion_net(disparity, prev_disp)
                except Exception as e:
                    logging.warning(f"Error in fusion net: {e}, skipping")

        disparity_right = None
        if self.cfg.USE_CONSISTENCY_LOSS:
            try:
                disparity_right = self.compute_disparity(right_image, left_image)
            except Exception as e:
                logging.warning(f"Error computing disparity_right: {e}, skipping")

        outputs = {"disparity": disparity, "disparity_right": disparity_right}

        disparity_next = None
        disparity_right_next = None
        if self.cfg.USE_TEMPORAL_LOSS and left_next is not None and right_next is not None:
            try:
                disparity_next = self.compute_disparity(left_next, right_next, disparity)
                outputs["disparity_next"] = disparity_next
                if self.cfg.USE_CONSISTENCY_LOSS:
                    disparity_right_next = self.compute_disparity(right_next, left_next)
                    outputs["disparity_right_next"] = disparity_right_next
            except Exception as e:
                logging.warning(f"Error computing disparity_next: {e}, skipping")

        if self.cfg.USE_DISP_COMPLETION and init_disp is not None:
            outputs["init_disp"] = init_disp
            outputs["completion_pred"] = completion_pred

        return outputs

    def inverse_warp(self, features, disp):
        if features is None:
            raise ValueError("inverse_warp: features tensor is None")
        if features.dim() != 4:
            raise ValueError(
                f"inverse_warp: Expected 4D features tensor, got {features.dim()}D tensor with shape {features.shape}")

        B, C, H, W = features.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=features.device),
                                            torch.arange(W, device=features.device), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=0).float()
        pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)
        disp = disp.squeeze(1)
        transformed_x = pixel_coords[:, 0, :, :] - disp
        grid = torch.stack([transformed_x, pixel_coords[:, 1, :, :]], dim=-1)
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)


# --- EvaluationMetrics ---
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
    def compute_temporal_consistency(disp, disp_next):
        return torch.mean((disp - disp_next) ** 2).item()

    @staticmethod
    def evaluate_reconstruction(inputs, loss_components, outputs):
        left_img = inputs["left_image"]
        warped_right = loss_components["warped_right_image"]
        metrics = {
            "psnr": EvaluationMetrics.compute_psnr(left_img, warped_right) if warped_right is not None else 0.0,
            "rmse": EvaluationMetrics.compute_rmse(left_img, warped_right) if warped_right is not None else 0.0,
            "ssim": EvaluationMetrics.compute_ssim(left_img, warped_right) if warped_right is not None else 0.0,
        }
        if "disparity_next" in outputs and outputs["disparity_next"] is not None:
            metrics["temporal_mse"] = EvaluationMetrics.compute_temporal_consistency(
                outputs["disparity"], outputs["disparity_next"])
        return metrics


# --- Trainer ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
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
            self.writer = SummaryWriter(
                log_dir=os.path.join(cfg.TENSORBOARD_DIR, datetime.now().strftime('%Y%m%d-%H%M%S')))

        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.optimizer = optim.AdamW([
            {'params': [p for p in self.model.cost_aggregator.parameters()], 'lr': cfg.LEARNING_RATE},
            {'params': [p for p in self.model.til_gru.parameters()], 'lr': cfg.LEARNING_RATE},
            {'params': [p for p in self.model.til_decoder.parameters()], 'lr': cfg.LEARNING_RATE},
            {'params': [p for p in self.model.completion_net.parameters()], 'lr': cfg.LEARNING_RATE},
            {'params': [p for p in self.model.fusion_net.parameters()], 'lr': cfg.LEARNING_RATE}
        ])
        self.loss_fn = ImprovedSelfSupervisedLoss(cfg)
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=5 * 224),
                optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.NUM_EPOCHS - 5, eta_min=1e-6)
            ],
            milestones=[5 * 224]
        )
        self.evaluator = EvaluationMetrics()

        train_dataset = RectifiedWaveStereoDataset(cfg, is_validation=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_fn, num_workers=4, pin_memory=self.device.type == 'cuda',
                                       drop_last=True)
        val_dataset = RectifiedWaveStereoDataset(cfg, is_validation=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=4, pin_memory=self.device.type == 'cuda',
                                     drop_last=True)

        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION and self.device.type == 'cuda')
        self.step = 0
        self.current_smoothness_weight = cfg.INITIAL_SMOOTHNESS_WEIGHT
        self.prev_disp = None

        self.loss_history = {
            'train': {'total': [], 'photometric': [], 'smoothness': [], 'consistency': [], 'temporal': [],
                      'grad_refine': [], 'disp_completion': []},
            'val': {'total': [], 'photometric': [], 'smoothness': [], 'consistency': [], 'temporal': [],
                    'grad_refine': [], 'disp_completion': []}}
        self.metric_history = {'train': {'psnr': [], 'rmse': [], 'ssim': [], 'temporal_mse': []},
                               'val': {'psnr': [], 'rmse': [], 'ssim': [], 'temporal_mse': []}}

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
        logging.info("\n--- 开始自监督训练 (优化版) ---")
        best_val_loss = float('inf')
        patience = 10
        epochs_no_improve = 0

        for epoch in range(self.cfg.NUM_EPOCHS):
            self.current_smoothness_weight *= self.cfg.SMOOTHNESS_WEIGHT_DECAY
            self.loss_fn.smoothness_weight = self.current_smoothness_weight

            self.model.train()
            epoch_train_loss_total = 0.0
            epoch_train_metrics = {'psnr': 0.0, 'rmse': 0.0, 'ssim': 0.0, 'temporal_mse': 0.0}
            last_train_loss_components = {}

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [训练]")

            for data in pbar:
                if data is None or data[0] is None:
                    continue
                if self.cfg.USE_TEMPORAL_LOSS:
                    left, right, mask, left_next, right_next, mask_next = [d.to(self.device) for d in data]
                    inputs = {"left_image": left, "right_image": right, "mask": mask,
                              "left_next": left_next, "right_next": right_next, "mask_next": mask_next}
                else:
                    left, right, mask = [d.to(self.device) for d in data]
                    inputs = {"left_image": left, "right_image": right, "mask": mask}

                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION and self.device.type == 'cuda'):
                    outputs = self.model(left, right, self.prev_disp,
                                         left_next=inputs.get("left_next"), right_next=inputs.get("right_next"))
                    loss_components = self.loss_fn(inputs, outputs)

                loss = loss_components["total_loss"]
                if not torch.isnan(loss):
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self.prev_disp = outputs["disparity"].detach() if "disparity" in outputs else None
                if self.prev_disp is not None and self.prev_disp.size(0) != self.cfg.BATCH_SIZE:
                    self.prev_disp = None

                step_metrics = self.evaluator.evaluate_reconstruction(inputs, loss_components, outputs)
                epoch_train_loss_total += loss.item() if not torch.isnan(loss) else 0.0
                for metric in epoch_train_metrics:
                    epoch_train_metrics[metric] += step_metrics.get(metric, 0.0)

                last_train_loss_components = {k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v for k, v in
                                              loss_components.items() if
                                              k != "warped_right_image" and k != "warped_left_image"}
                logging.debug(f"Loss components: {last_train_loss_components}")

                if self.writer:
                    self.writer.add_scalar('Loss/train/total', loss.item(), self.step)
                    for k, v in last_train_loss_components.items():
                        if isinstance(v, float) and k != "total_loss":
                            self.writer.add_scalar(f'Loss/train/{k}', v, self.step)
                    for k, v in step_metrics.items():
                        self.writer.add_scalar(f'Metrics/train/{k}', v, self.step)
                    self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], self.step)

                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(inputs, outputs, loss_components, step_metrics, mode='train')

                self.step += 1
                pbar.set_postfix({
                    'loss': f"{loss.item():.3f}" if not torch.isnan(loss) else "NaN",
                    'psnr': f"{step_metrics.get('psnr', 0):.1f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

            num_batches = len(self.train_loader)
            if num_batches > 0:
                epoch_train_loss_total /= num_batches
                for metric in epoch_train_metrics:
                    epoch_train_metrics[metric] /= num_batches
                self.loss_history['train']['total'].append(epoch_train_loss_total)
                for k in ['photometric', 'smoothness', 'consistency', 'temporal', 'grad_refine', 'disp_completion']:
                    v = last_train_loss_components.get(k + '_loss', 0.0)
                    if k not in self.loss_history['train']:
                        logging.warning(
                            f"Loss component '{k}' not found in loss_history['train'], initializing with 0.0")
                        self.loss_history['train'][k] = [0.0] * len(self.loss_history['train']['total'])
                    self.loss_history['train'][k].append(v if isinstance(v, float) else 0.0)
                for k, v in epoch_train_metrics.items():
                    self.metric_history['train'][k].append(v)

            avg_val_loss, val_metrics = self.validate(epoch)
            logging.info(
                f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> 验证损失: {avg_val_loss:.4f}, PSNR: {val_metrics.get('psnr', 0):.2f}")

            if not np.isnan(avg_val_loss):
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model_self_supervised_optimized.pth")
                    torch.save(self.model.state_dict(), save_path)
                    logging.info(f"✓ 验证损失降低，模型已保存至: {save_path}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logging.info(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss")
                        break

            self.scheduler.step()
            self.update_log_file(epoch)
            if self.cfg.VISUALIZE_TRAINING:
                self.plot_training_history()

        logging.info("训练完成!")
        if self.writer:
            self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        epoch_val_loss_total = 0.0
        epoch_val_metrics = {'psnr': 0.0, 'rmse': 0.0, 'ssim': 0.0, 'temporal_mse': 0.0}
        num_val_batches = 0
        last_val_loss_components = {}
        val_pbar = tqdm(self.val_loader, desc="[验证]")

        with torch.no_grad():
            for data in val_pbar:
                if data is None or data[0] is None:
                    continue
                if self.cfg.USE_TEMPORAL_LOSS:
                    left, right, mask, left_next, right_next, mask_next = [d.to(self.device) for d in data]
                    inputs = {"left_image": left, "right_image": right, "mask": mask,
                              "left_next": left_next, "right_next": right_next, "mask_next": mask_next}
                else:
                    left, right, mask = [d.to(self.device) for d in data]
                    inputs = {"left_image": left, "right_image": right, "mask": mask}

                with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION and self.device.type == 'cuda'):
                    outputs = self.model(left, right, self.prev_disp,
                                         left_next=inputs.get("left_next"), right_next=inputs.get("right_next"))
                    loss_components = self.loss_fn(inputs, outputs)

                loss = loss_components["total_loss"]
                if not torch.isnan(loss):
                    epoch_val_loss_total += loss.item()
                    num_val_batches += 1

                step_metrics = self.evaluator.evaluate_reconstruction(inputs, loss_components, outputs)
                for metric in epoch_val_metrics:
                    epoch_val_metrics[metric] += step_metrics.get(metric, 0.0)

                last_val_loss_components = {k: v.item() if torch.is_tensor(v) and v.numel() == 1 else v for k, v in
                                            loss_components.items() if
                                            k != "warped_right_image" and k != "warped_left_image"}
                logging.debug(f"Validation loss components: {last_val_loss_components}")

                if self.writer:
                    self.writer.add_scalar('Loss/val/total', loss.item(), self.step)
                    for k, v in last_val_loss_components.items():
                        if isinstance(v, float) and k != "total_loss":
                            self.writer.add_scalar(f'Loss/val/{k}', v, self.step)
                    for k, v in step_metrics.items():
                        self.writer.add_scalar(f'Metrics/val/{k}', v, self.step)

                if self.cfg.VISUALIZE_TRAINING and num_val_batches == 1:
                    self.visualize(inputs, outputs, loss_components, step_metrics, mode='val')

                val_pbar.set_postfix({
                    'loss': f"{loss.item():.3f}" if not torch.isnan(loss) else "NaN",
                    'psnr': f"{step_metrics.get('psnr', 0):.1f}"
                })

        if num_val_batches > 0:
            epoch_val_loss_total /= num_val_batches
            for metric in epoch_val_metrics:
                epoch_val_metrics[metric] /= num_val_batches
            self.loss_history['val']['total'].append(epoch_val_loss_total)
            for k in ['photometric', 'smoothness', 'consistency', 'temporal', 'grad_refine', 'disp_completion']:
                v = last_val_loss_components.get(k + '_loss', 0.0)
                if k not in self.loss_history['val']:
                    logging.warning(f"Loss component '{k}' not found in loss_history['val'], initializing with 0.0")
                    self.loss_history['val'][k] = [0.0] * len(self.loss_history['val']['total'])
                self.loss_history['val'][k].append(v if isinstance(v, float) else 0.0)
            for k, v in epoch_val_metrics.items():
                self.metric_history['val'][k].append(v)

        return epoch_val_loss_total, epoch_val_metrics

    def visualize(self, inputs, outputs, loss_components, metrics, mode='train'):
        left_img = inputs["left_image"][0].detach().cpu().permute(1, 2, 0).numpy()
        right_img = inputs["right_image"][0].detach().cpu().permute(1, 2, 0).numpy()
        warped_right = loss_components["warped_right_image"][0].detach().cpu().permute(1, 2, 0).numpy() if \
        loss_components["warped_right_image"] is not None else np.zeros_like(left_img)
        pred_disp = outputs["disparity"][0, 0].detach().cpu().numpy()

        fig = plt.figure(figsize=(15, 10))
        plt.subplot(3, 2, 1)
        plt.imshow(left_img, cmap='gray')
        plt.title("左图")
        plt.axis('off')

        plt.subplot(3, 2, 2)
        plt.imshow(right_img, cmap='gray')
        plt.title("右图")
        plt.axis('off')

        plt.subplot(3, 2, 3)
        plt.imshow(warped_right, cmap='gray')
        plt.title(f"右图经视差变换 (PSNR: {metrics.get('psnr', 0):.2f})")
        plt.axis('off')

        plt.subplot(3, 2, 4)
        im4 = plt.imshow(pred_disp, cmap='jet')
        plt.title("预测视差图")
        plt.axis('off')
        fig.colorbar(im4, fraction=0.046, pad=0.04)

        if "disparity_right" in outputs and outputs["disparity_right"] is not None:
            pred_disp_right = outputs["disparity_right"][0, 0].detach().cpu().numpy()
            plt.subplot(3, 2, 5)
            im5 = plt.imshow(pred_disp_right, cmap='jet')
            plt.title("右视差图")
            plt.axis('off')
            fig.colorbar(im5, fraction=0.046, pad=0.04)

        if "disparity_next" in outputs and outputs["disparity_next"] is not None:
            pred_disp_next = outputs["disparity_next"][0, 0].detach().cpu().numpy()
            plt.subplot(3, 2, 6)
            diff_disp = np.abs(pred_disp - pred_disp_next)
            im6 = plt.imshow(diff_disp, cmap='hot')
            plt.title("帧间视差差值")
            plt.axis('off')
            fig.colorbar(im6, fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, f"{mode}_epoch_{self.step}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logging.info(f"✓ 可视化图像已保存至: {save_path}")

    def plot_training_history(self):
        fig = plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history['train']['total'], label='训练总损失')
        plt.plot(self.loss_history['val']['total'], label='验证总损失')
        plt.title("总损失")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.metric_history['train']['psnr'], label='训练PSNR')
        plt.plot(self.metric_history['val']['psnr'], label='验证PSNR')
        plt.title("PSNR")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(self.metric_history['train']['ssim'], label='训练SSIM')
        plt.plot(self.metric_history['val']['ssim'], label='验证SSIM')
        plt.title("SSIM")
        plt.xlabel("Epoch")
        plt.ylabel("SSIM")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(self.metric_history['train']['temporal_mse'], label='训练时序MSE')
        plt.plot(self.metric_history['val']['temporal_mse'], label='验证时序MSE')
        plt.title("时序一致性MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR,
                                 f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logging.info(f"✓ 训练历史图已保存至: {save_path}")

    def update_log_file(self, epoch):
        log_data = {
            'epoch': epoch + 1,
            'train_loss': {k: v[-1] if v else 0.0 for k, v in self.loss_history['train'].items()},
            'val_loss': {k: v[-1] if v else 0.0 for k, v in self.loss_history['val'].items()},
            'train_metrics': {k: v[-1] if v else 0.0 for k, v in self.metric_history['train'].items()},
            'val_metrics': {k: v[-1] if v else 0.0 for k, v in self.metric_history['val'].items()},
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        with open(self.log_file, 'a') as f:
            json.dump(log_data, f, indent=4)
            f.write('\n')


if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()
