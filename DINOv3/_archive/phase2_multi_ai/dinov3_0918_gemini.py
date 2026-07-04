# self_supervised_dinov3_temporal.py
# An end-to-end, self-supervised 3D reconstruction pipeline for dynamic wave surfaces.
# This version integrates temporal consistency, advanced data augmentation, and smarter training strategies.
# FIX: Corrected CUDA launch failure by fixing dtype mismatch, improving DataLoader stability, and adding robust data loading.

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
from math import exp

# --- TensorBoard Import ---
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("[WARNING]: Could not import SummaryWriter. TensorBoard monitoring will be unavailable.")
    print("Please install it: pip install tensorboard")
    SummaryWriter = None

# --- Hugging Face Transformers Import ---
try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80)
    print("[FATAL ERROR]: Could not import 'AutoModel' from the 'transformers' library.")
    print("Please ensure 'transformers' is installed: pip install transformers")
    print("For better performance, also install 'accelerate': pip install accelerate")
    print("=" * 80)
    sys.exit(1)

# --- 1. Configuration Center ---

# Absolute path to the project root directory
PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


@dataclass
class Config:
    """Project Configuration Parameters"""
    # --- File Paths ---
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")
    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs")
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")

    # --- Visualization Control ---
    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 100

    # --- Data Processing Parameters ---
    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 512
    MASK_THRESHOLD: int = 30

    # --- Model & Training Parameters ---
    BATCH_SIZE: int = 2  # Reduced batch size for temporal data
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    MAX_DISPARITY: int = 128

    # --- Optimization Parameters ---
    USE_MIXED_PRECISION: bool = True
    PHOTOMETRIC_LOSS_WEIGHTS: tuple = (0.85, 0.15)  # SSIM, L1
    USE_CONSISTENCY_LOSS: bool = True
    CONSISTENCY_LOSS_WEIGHT: float = 0.1
    INITIAL_SMOOTHNESS_WEIGHT: float = 0.5
    SMOOTHNESS_WEIGHT_DECAY: float = 0.98

    # --- [NEW] Advanced Data Augmentation ---
    USE_ADVANCED_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.5

    # --- [NEW] Temporal Consistency Parameters ---
    USE_TEMPORAL_LOSS: bool = True
    TEMPORAL_LOSS_WEIGHT: float = 0.2

    # --- [NEW] Advanced Training Strategies ---
    EARLY_STOPPING_PATIENCE: int = 10
    WARMUP_EPOCHS: int = 5


# --- 2. Self-Supervised Loss Function ---
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.register_buffer('weight', self.gaussian(11, 1.5))
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / (SSIM_d + 1e-8)) / 2, 0, 1)


class ImprovedSelfSupervisedLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.smoothness_weight = cfg.INITIAL_SMOOTHNESS_WEIGHT
        self.photometric_weights = cfg.PHOTOMETRIC_LOSS_WEIGHTS
        self.consistency_weight = cfg.CONSISTENCY_LOSS_WEIGHT
        self.temporal_weight = cfg.TEMPORAL_LOSS_WEIGHT
        self.ssim = SSIM()

    def compute_photometric_loss(self, img1, img2, pred_disp, mask):
        warped_img2 = self.inverse_warp(img2, pred_disp)
        mask_sum = mask.sum() + 1e-8

        # L1 Loss
        l1_map = torch.abs(warped_img2 - img1)
        l1_loss = (l1_map * mask).sum() / mask_sum

        # SSIM Loss
        ssim_map = self.ssim(warped_img2, img1)
        ssim_loss = (ssim_map * mask).sum() / mask_sum

        photometric_loss = self.photometric_weights[0] * ssim_loss + self.photometric_weights[1] * l1_loss
        return photometric_loss, warped_img2

    def forward(self, inputs, outputs):
        left_img, right_img = inputs["left_image"], inputs["right_image"]
        mask = inputs["mask"]
        pred_disp = outputs["disparity"]

        # Photometric loss for current frame (t)
        photometric_loss_right, warped_right_image = self.compute_photometric_loss(left_img, right_img, pred_disp, mask)
        photometric_loss_left, warped_left_image = self.compute_photometric_loss(right_img, left_img, -pred_disp, mask)
        photometric_loss = (photometric_loss_right + photometric_loss_left) / 2.0

        # Smoothness loss
        smoothness_loss = self.compute_smoothness_loss(pred_disp, left_img, mask)

        # Consistency loss
        consistency_loss = torch.tensor(0.0, device=left_img.device)
        if self.cfg.USE_CONSISTENCY_LOSS and "disparity_right" in outputs and outputs["disparity_right"] is not None:
            disp_left, disp_right = outputs["disparity"], outputs["disparity_right"]
            warped_disp_right = self.inverse_warp(disp_right, -disp_left)
            consistency_loss = (torch.abs(disp_left - warped_disp_right) * mask).sum() / (mask.sum() + 1e-8)

        # Temporal loss
        temporal_loss = torch.tensor(0.0, device=left_img.device)
        if self.cfg.USE_TEMPORAL_LOSS and "disparity_next" in outputs and outputs["disparity_next"] is not None:
            pred_disp_next = outputs["disparity_next"]
            mask_next = inputs.get("mask_next", mask)  # Use current mask if next is unavailable
            temporal_loss = (torch.abs(pred_disp - pred_disp_next) * mask_next).sum() / (mask_next.sum() + 1e-8)

        total_loss = photometric_loss + self.smoothness_weight * smoothness_loss
        total_loss += self.consistency_weight * consistency_loss
        total_loss += self.temporal_weight * temporal_loss

        return {
            "total_loss": total_loss,
            "photometric_loss": photometric_loss,
            "smoothness_loss": smoothness_loss,
            "consistency_loss": consistency_loss,
            "temporal_loss": temporal_loss,
            "warped_right_image": warped_right_image,
            "warped_left_image": warped_left_image,
        }

    def inverse_warp(self, features, disp):
        B, C, H, W = features.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=features.device),
                                            torch.arange(W, device=features.device), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=0).float().repeat(B, 1, 1, 1)
        disp_unsq = disp.squeeze(1)
        transformed_x = pixel_coords[:, 0, :, :] - disp_unsq
        grid = torch.stack([transformed_x, pixel_coords[:, 1, :, :]], dim=-1)
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def compute_smoothness_loss(self, disp, img, mask):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        mask_x, mask_y = mask[:, :, :, :-1], mask[:, :, :-1, :]
        smoothness_x = (grad_disp_x * mask_x).sum() / (mask_x.sum() + 1e-8)
        smoothness_y = (grad_disp_y * mask_y).sum() / (mask_y.sum() + 1e-8)
        return smoothness_x + smoothness_y


# --- 3. PyTorch Dataset ---
class RectifiedWaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.is_validation = is_validation
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"[FATAL ERROR]: No image files found in '{cfg.LEFT_IMAGE_DIR}'.")

        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            self.roi_left, self.roi_right = tuple(calib['roi_left']), tuple(calib['roi_right'])
        except Exception as e:
            sys.exit(f"[FATAL ERROR]: Failed to load calibration file: {e}")

        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

        # If using temporal loss, remove the last index to ensure a pair exists
        if cfg.USE_TEMPORAL_LOSS:
            self.indices = self.indices[:-1]

    def __len__(self):
        return len(self.indices)

    def _load_and_process_frame(self, frame_idx):
        left_img_path = self.left_images[frame_idx]
        frame_basename = os.path.basename(left_img_path)
        right_frame_basename = 'right' + frame_basename[4:] if frame_basename.startswith('left') else frame_basename
        right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_frame_basename)

        if not os.path.exists(right_img_path): return None
        left_raw = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        right_raw = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
        if left_raw is None or right_raw is None: return None

        left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

        x, y, w, h = self.roi_left
        left_rect = left_rect[y:y + h, x:x + w]
        x, y, w, h = self.roi_right
        right_rect = right_rect[y:y + h, x:x + w]

        target_h, target_w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
        left_img = cv2.resize(left_rect, (target_w, target_h))
        right_img = cv2.resize(right_rect, (target_w, target_h))

        _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

        return left_img, right_img, mask

    def _apply_augmentation(self, left, right):
        # Photometric augmentations
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            left = np.clip(left * brightness, 0, 255)
            right = np.clip(right * brightness, 0, 255)
        if np.random.rand() < 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            left_mean, right_mean = left.mean(), right.mean()
            left = np.clip((left - left_mean) * contrast + left_mean, 0, 255)
            right = np.clip((right - right_mean) * contrast + right_mean, 0, 255)

        # [NEW] Noise augmentation
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 10, left.shape)
            left = np.clip(left + noise, 0, 255)
            right = np.clip(right + noise, 0, 255)

        # [NEW] Geometric augmentation (flip)
        if np.random.rand() < 0.5:
            left = cv2.flip(left, 1)
            right = cv2.flip(right, 1)

        return left.astype(np.uint8), right.astype(np.uint8)

    def __getitem__(self, idx):
        try:
            # Load current frame
            current_frame_idx = self.indices[idx]
            processed_current = self._load_and_process_frame(current_frame_idx)
            if processed_current is None: return None
            left_img, right_img, mask = processed_current

            # Load next frame if temporal loss is enabled
            if self.cfg.USE_TEMPORAL_LOSS:
                next_frame_idx = self.indices[idx + 1] if idx + 1 < len(self.indices) else current_frame_idx
                processed_next = self._load_and_process_frame(next_frame_idx)
                if processed_next is None:
                    # If next frame fails, duplicate current frame to maintain batch size
                    left_next, right_next, mask_next = left_img.copy(), right_img.copy(), mask.copy()
                else:
                    left_next, right_next, mask_next = processed_next

            # Apply synchronized augmentations
            if not self.is_validation and self.cfg.USE_ADVANCED_AUGMENTATION and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
                left_img, right_img = self._apply_augmentation(left_img, right_img)
                if self.cfg.USE_TEMPORAL_LOSS:
                    left_next, right_next = self._apply_augmentation(left_next, right_next)

            # Convert to BGR and then to Tensor
            def to_tensor(img):
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return torch.from_numpy(img_bgr.transpose(2, 0, 1)).float() / 255.0

            left_tensor = to_tensor(left_img)
            right_tensor = to_tensor(right_img)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

            if self.cfg.USE_TEMPORAL_LOSS:
                left_next_tensor = to_tensor(left_next)
                right_next_tensor = to_tensor(right_next)
                mask_next_tensor = torch.from_numpy(mask_next).float().unsqueeze(0) / 255.0
                return left_tensor, right_tensor, mask_tensor, left_next_tensor, right_next_tensor, mask_next_tensor
            else:
                return left_tensor, right_tensor, mask_tensor
        except Exception as e:
            print(f"Warning: Error processing data at index {idx}: {e}. Skipping sample.")
            return None


# --- 4. Deep Learning Model ---
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
        if self.dino is None: sys.exit(1)

        for param in self.dino.parameters():
            param.requires_grad = False

        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)
        self.cost_aggregator = nn.Sequential(
            conv_block_3d(self.feature_dim, 32),
            Hourglass3D(32),
            nn.Conv3d(32, 1, 3, padding=1)
        )
        print("✓ Model built (3D Hourglass Architecture).")

    def _load_dino_model(self):
        local_path = self.cfg.DINO_LOCAL_PATH
        print(f"--- Attempting to load DINOv3 model from: {local_path} ---")
        if not os.path.isdir(local_path):
            print(f"[FATAL ERROR]: Model directory not found: '{local_path}'")
            return None
        try:
            model = AutoModel.from_pretrained(local_path, local_files_only=True)
            print(f"✓ Successfully loaded DINOv3 model from local path.")
            return model
        except Exception as e:
            print(f"[FATAL ERROR]: Error loading model from '{local_path}': {e}")
            return None

    def _compute_disparity(self, left_feat, right_feat, h, w):
        cost_volume = self.build_cost_volume(left_feat, right_feat)
        cost_aggregated = self.cost_aggregator(cost_volume).squeeze(1)
        cost_softmax = F.softmax(-cost_aggregated, dim=1)
        max_disp_feat = self.max_disp // self.patch_size
        disp_values = torch.arange(0, max_disp_feat, device=cost_softmax.device, dtype=torch.float32).view(1, -1, 1, 1)
        disparity_feat = torch.sum(cost_softmax * disp_values, 1, keepdim=True)
        disparity = F.interpolate(disparity_feat * self.patch_size, size=(h, w), mode='bilinear', align_corners=False)
        return disparity

    def get_features(self, image):
        b, c, h, w = image.shape
        with torch.no_grad():
            outputs = self.dino(image, output_hidden_states=False)
            features = outputs.last_hidden_state
        start_index = 1 + self.num_register_tokens
        patch_tokens = features[:, start_index:, :]
        feature_h, feature_w = h // self.patch_size, w // self.patch_size
        features_2d = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, feature_h, feature_w)
        return features_2d

    def build_cost_volume(self, left_feat, right_feat):
        B, C, H, W = left_feat.shape
        max_disp_feat = self.max_disp // self.patch_size
        # [FIX] Explicitly set dtype to match input features for mixed-precision safety
        cost_volume = torch.zeros(B, C, max_disp_feat, H, W, device=left_feat.device, dtype=left_feat.dtype)
        for d in range(max_disp_feat):
            if d > 0:
                cost_volume[:, :, d, :, d:] = left_feat[:, :, :, d:] - right_feat[:, :, :, :-d]
            else:
                cost_volume[:, :, d, :, :] = left_feat - right_feat
        return cost_volume

    def forward(self, left_image, right_image, left_next=None, right_next=None):
        h, w = left_image.shape[-2:]
        left_feat = self.get_features(left_image)
        right_feat = self.get_features(right_image)
        disparity = self._compute_disparity(left_feat, right_feat, h, w)

        outputs = {"disparity": disparity}

        if self.cfg.USE_CONSISTENCY_LOSS:
            outputs["disparity_right"] = self._compute_disparity(right_feat, left_feat, h, w)

        if self.cfg.USE_TEMPORAL_LOSS and left_next is not None and right_next is not None:
            left_feat_next = self.get_features(left_next)
            right_feat_next = self.get_features(right_next)
            outputs["disparity_next"] = self._compute_disparity(left_feat_next, right_feat_next, h, w)

        return outputs


# --- 5. Evaluation Metrics ---
class EvaluationMetrics:
    @staticmethod
    def compute_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    @staticmethod
    def compute_rmse(img1, img2):
        return torch.sqrt(torch.mean((img1 - img2) ** 2)).item()

    @staticmethod
    def compute_temporal_consistency(disp, disp_next):
        return torch.mean((disp - disp_next) ** 2).item()

    @staticmethod
    def evaluate_reconstruction(inputs, outputs, loss_components):
        left_img = inputs["left_image"]
        warped_right = loss_components["warped_right_image"]
        metrics = {
            "psnr": EvaluationMetrics.compute_psnr(left_img, warped_right),
            "rmse": EvaluationMetrics.compute_rmse(left_img, warped_right),
        }
        if "disparity_next" in outputs and outputs["disparity_next"] is not None:
            metrics["temporal_mse"] = EvaluationMetrics.compute_temporal_consistency(
                outputs["disparity"], outputs["disparity_next"])
        return metrics


# --- 6. Trainer ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR, self.timestamp)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.visualization_dir = os.path.join(self.run_dir, "visualizations")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")
        for d in [self.checkpoint_dir, self.visualization_dir, self.log_dir, self.tensorboard_dir]:
            os.makedirs(d, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {self.device}")
        if self.device.type == 'cpu': cfg.USE_MIXED_PRECISION = False

        self.writer = SummaryWriter(log_dir=self.tensorboard_dir) if SummaryWriter else None
        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.loss_fn = ImprovedSelfSupervisedLoss(cfg)

        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS,
                                                              eta_min=1e-6)
        warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                                                       total_iters=cfg.WARMUP_EPOCHS)
        self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
                                                         milestones=[cfg.WARMUP_EPOCHS])

        self.evaluator = EvaluationMetrics()
        train_dataset = RectifiedWaveStereoDataset(cfg, is_validation=False)
        val_dataset = RectifiedWaveStereoDataset(cfg, is_validation=True)
        # [FIX] Set num_workers to 0 for better stability, especially on Windows
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                                       num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=0, pin_memory=True)

        # [FIX] Updated to modern torch.amp API
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION)
        self.step = 0
        self.current_smoothness_weight = cfg.INITIAL_SMOOTHNESS_WEIGHT
        self.log_file = os.path.join(self.log_dir, "training_log.json")
        self.history = self._init_history()
        self.loss_keys = ['total', 'photometric', 'smoothness', 'consistency', 'temporal']
        self.metric_keys = ['psnr', 'rmse', 'temporal_mse']

    def _init_history(self):
        loss_keys = ['total', 'photometric', 'smoothness', 'consistency', 'temporal']
        metric_keys = ['psnr', 'rmse', 'temporal_mse']
        history = {
            'train': {k: [] for k in loss_keys + metric_keys},
            'val': {k: [] for k in loss_keys + metric_keys}
        }
        return history

    def train(self):
        print("\n--- Starting Self-Supervised Training (Optimized) ---")
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.cfg.NUM_EPOCHS):
            self.current_smoothness_weight *= self.cfg.SMOOTHNESS_WEIGHT_DECAY
            self.loss_fn.smoothness_weight = self.current_smoothness_weight

            train_results = self._run_epoch(epoch, is_training=True)
            self._log_epoch_results('train', epoch, train_results)

            val_results = self._run_epoch(epoch, is_training=False)
            self._log_epoch_results('val', epoch, val_results)

            avg_val_loss = val_results['total']
            print(
                f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> Train Loss: {train_results['total']:.4f} | Val Loss: {avg_val_loss:.4f}, Val PSNR: {val_results.get('psnr', 0):.2f}")

            self.scheduler.step()
            self.update_log_file(epoch)
            if self.cfg.VISUALIZE_TRAINING: self.plot_training_history()

            if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                save_path = os.path.join(self.checkpoint_dir, "best_model_temporal.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ Val loss improved. Model saved to: {save_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.EARLY_STOPPING_PATIENCE:
                    print(f"--- Early stopping triggered after {epoch + 1} epochs. ---")
                    break

        print("--- Training complete! ---")
        if self.writer: self.writer.close()

    def _run_epoch(self, epoch, is_training):
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        phase = "Training" if is_training else "Validation"
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [{phase}]")

        epoch_losses = {k: 0.0 for k in self.loss_keys}
        epoch_metrics = {k: 0.0 for k in self.metric_keys}

        for data in pbar:
            if data is None: continue

            if self.cfg.USE_TEMPORAL_LOSS:
                left, right, mask, left_next, right_next, mask_next = [d.to(self.device) for d in data]
                inputs = {"left_image": left, "right_image": right, "mask": mask, "left_next": left_next,
                          "right_next": right_next, "mask_next": mask_next}
            else:
                left, right, mask = [d.to(self.device) for d in data]
                inputs = {"left_image": left, "right_image": right, "mask": mask}

            with torch.set_grad_enabled(is_training):
                # [FIX] Updated to modern torch.amp API
                with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION):
                    outputs = self.model(left, right, inputs.get("left_next"), inputs.get("right_next"))
                    loss_components = self.loss_fn(inputs, outputs)
                    loss = loss_components["total_loss"]

                if is_training and not (torch.isnan(loss) or torch.isinf(loss)):
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            step_metrics = self.evaluator.evaluate_reconstruction(inputs, outputs, loss_components)

            # [FIX] Correctly map history keys to loss_component keys
            for k in epoch_losses: epoch_losses[k] += loss_components.get(f"{k}_loss", torch.tensor(0.0)).item()
            for k in epoch_metrics: epoch_metrics[k] += step_metrics.get(k, 0.0)

            pbar.set_postfix({'loss': loss.item(), 'psnr': step_metrics.get('psnr', 0)})

            if is_training:
                if self.writer:
                    self.writer.add_scalar(f'Loss/step_train', loss.item(), self.step)
                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(inputs, outputs, loss_components, self.step, "train")
                self.step += 1

        num_batches = len(loader)
        if num_batches == 0: return {**{k: 0.0 for k in self.loss_keys}, **{k: 0.0 for k in self.metric_keys}}

        results = {k: v / num_batches for k, v in {**epoch_losses, **epoch_metrics}.items()}
        return results

    def _log_epoch_results(self, phase, epoch, results):
        for k, v in results.items():
            if k in self.history[phase]:
                self.history[phase][k].append(v)
            if self.writer:
                key_type = 'Loss' if k in self.loss_keys else 'Metrics'
                self.writer.add_scalar(f'{key_type}/{phase}_{k}', v, epoch)

        # [FIX] Correctly use keys from the results dict and check for history existence
        if self.writer and phase == 'val' and self.history['train']['total']:
            self.writer.add_scalars('Loss/epoch_comparison',
                                    {'train': self.history['train']['total'][-1], 'val': results['total']},
                                    epoch)
            if 'psnr' in self.history['train'] and self.history['train']['psnr']:
                self.writer.add_scalars('PSNR/epoch_comparison',
                                        {'train': self.history['train']['psnr'][-1], 'val': results['psnr']},
                                        epoch)

    def visualize(self, inputs, outputs, loss_components, step, phase="train"):
        left_img = (inputs["left_image"][0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pred_disp = outputs["disparity"][0, 0].cpu().detach().numpy()
        mask = inputs["mask"][0, 0].cpu().numpy()
        masked_disp = np.ma.masked_where(mask == 0, pred_disp)
        warped_right = (loss_components["warped_right_image"][0].cpu().permute(1, 2, 0).detach().numpy() * 255).astype(
            np.uint8)

        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        plt.suptitle(f'Visualization - Step: {step} ({phase})', fontsize=16)

        axes[0, 0].imshow(left_img)
        axes[0, 0].set_title("Input Left Image")
        axes[0, 1].imshow(warped_right)
        axes[0, 1].set_title("Reconstructed Left Image")

        im2 = axes[1, 0].imshow(masked_disp, cmap='viridis')
        axes[1, 0].set_title("Predicted Disparity (Masked)")
        axes[1, 0].set_facecolor('black')
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        diff_map = np.abs(left_img.astype(float) - warped_right.astype(float)).mean(axis=2)
        im3 = axes[1, 1].imshow(diff_map, cmap='hot')
        axes[1, 1].set_title("Photometric Error Map")
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        axes[2, 0].imshow(mask, cmap='gray')
        axes[2, 0].set_title("Attention Mask")

        if "disparity_next" in outputs and outputs["disparity_next"] is not None:
            pred_disp_next = outputs["disparity_next"][0, 0].cpu().detach().numpy()
            disp_diff = np.abs(pred_disp - pred_disp_next)
            im6 = axes[2, 1].imshow(disp_diff, cmap='hot')
            axes[2, 1].set_title("Temporal Disparity Difference")
            fig.colorbar(im6, ax=axes[2, 1], fraction=0.046, pad=0.04)
        else:
            axes[2, 1].axis('off')

        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.visualization_dir, f"{phase}_step_{step:06d}.png")
        plt.savefig(save_path)
        if self.writer: self.writer.add_figure(f'Visualization/{phase}', fig, global_step=step)
        plt.close(fig)

    def plot_training_history(self):
        if not self.history['train']['total']: return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)

        # Plot Total Loss
        axes[0, 0].plot(self.history['train']['total'], label='Train Loss')
        axes[0, 0].plot(self.history['val']['total'], label='Val Loss')
        axes[0, 0].set_title('Total Loss');
        axes[0, 0].legend();
        axes[0, 0].grid(True)

        # Plot Loss Components
        axes[0, 1].plot(self.history['train']['photometric'], label='Photometric')
        axes[0, 1].plot(self.history['train']['smoothness'], label='Smoothness')
        axes[0, 1].plot(self.history['train']['temporal'], label='Temporal')
        axes[0, 1].set_title('Loss Components (Train)');
        axes[0, 1].legend();
        axes[0, 1].grid(True)

        # Plot PSNR
        axes[1, 0].plot(self.history['train']['psnr'], label='Train PSNR')
        axes[1, 0].plot(self.history['val']['psnr'], label='Val PSNR')
        axes[1, 0].set_title('PSNR');
        axes[1, 0].legend();
        axes[1, 0].grid(True)

        # Plot Temporal MSE
        axes[1, 1].plot(self.history['train']['temporal_mse'], label='Train Temporal MSE')
        axes[1, 1].plot(self.history['val']['temporal_mse'], label='Val Temporal MSE')
        axes[1, 1].set_title('Temporal Consistency MSE');
        axes[1, 1].legend();
        axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.visualization_dir, "training_history.png")
        plt.savefig(save_path)
        plt.close(fig)

    def update_log_file(self, epoch):
        log_data = {
            'config': {k: str(v) for k, v in asdict(self.cfg).items()},
            'epoch': epoch,
            'history': self.history,
            'update_time': datetime.now().isoformat()
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)


# --- 7. Main Execution Block ---
if __name__ == '__main__':
    if SummaryWriter is None:
        sys.exit("[ERROR]: TensorBoard is not installed. Please run 'pip install tensorboard'.")
    plt.switch_backend('Agg')  # Use 'Agg' backend to avoid GUI errors on headless servers
    config = Config()
    trainer = Trainer(config)
    trainer.train()

