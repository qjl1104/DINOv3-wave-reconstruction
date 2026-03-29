# self_supervised_dinov3_raft_temporal.py
# Temporal enhancement: Using 3 consecutive frames for motion prior
# Key idea: Temporal consistency helps constrain sparse feature matching

import os
import sys
import glob
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import subprocess

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80 + "\n[FATAL ERROR]: 'transformers' library not found.\n" + "=" * 80)
    sys.exit(1)

PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


@dataclass
class Config:
    """Configuration with temporal settings"""
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")
    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs_raft_temporal")
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")

    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 100
    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 512
    MASK_THRESHOLD: int = 30

    # Temporal settings
    TEMPORAL_FRAMES: int = 3  # Use 3 consecutive frames
    TEMPORAL_WEIGHT: float = 0.1  # Weight for temporal consistency loss

    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    GRADIENT_ACCUMULATION_STEPS: int = 1

    USE_MIXED_PRECISION: bool = True
    PHOTOMETRIC_LOSS_WEIGHTS: tuple = (0.85, 0.15)
    SMOOTHNESS_WEIGHT: float = 0.05

    USE_ADVANCED_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.8

    EARLY_STOPPING_PATIENCE: int = 25

    ITERATIONS: int = 8  # Back to 8 iterations
    DEEP_SUPERVISION_DECAY: float = 0.8
    DISP_UPDATE_SCALE: float = 1.0


def auto_tune_config(cfg: Config):
    print("--- Probing GPU for auto-tuning ---")
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      stderr=subprocess.DEVNULL).decode("utf-8").strip()
        free_mem_mb = int(out)
    except Exception:
        free_mem_mb = None

    patch_size = 16
    if not free_mem_mb:
        print("[WARNING] Could not probe GPU. Using conservative settings.")
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT = 1, 6, 384, 192
        return cfg

    print(f"Available GPU Memory: {free_mem_mb} MB")

    # Temporal processing requires more memory
    if free_mem_mb < 8000:
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 6, 4, 0.7
    elif free_mem_mb < 12000:
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 8, 2, 0.8
    elif free_mem_mb < 20000:
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 8, 1, 0.9
    else:
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 2, 8, 1, 1.0

    cfg.IMAGE_WIDTH = int((cfg.IMAGE_WIDTH * scale) // patch_size) * patch_size
    cfg.IMAGE_HEIGHT = int((cfg.IMAGE_HEIGHT * scale) // patch_size) * patch_size

    print(f"  Resolution: {cfg.IMAGE_WIDTH}x{cfg.IMAGE_HEIGHT}, Batch: {cfg.BATCH_SIZE}, "
          f"Temporal Frames: {cfg.TEMPORAL_FRAMES}, Iterations: {cfg.ITERATIONS}")
    return cfg


# --- Core Model Components ---
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], 1)
        z, r = torch.sigmoid(self.convz(hx)), torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], 1)))
        return (1 - z) * h + z * q


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        corr_ch = (2 * 4 + 1) ** 2 * 4
        self.conv1 = nn.Sequential(nn.Conv2d(2 + corr_ch, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.ReLU(True))

    def forward(self, disp, corr):
        x = torch.cat([disp, corr], 1)
        return self.conv3(self.conv2(self.conv1(x)))


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout_prob=0.1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout2d(dropout_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.GroupNorm(8, out_ch))

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class TemporalFusion(nn.Module):
    """Fuse features from multiple temporal frames"""

    def __init__(self, feature_dim):
        super(TemporalFusion, self).__init__()
        # Simple 1x1 conv to fuse temporal features
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, 1),
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(True)
        )

    def forward(self, feat_prev, feat_curr, feat_next):
        """
        Args:
            feat_prev: Features from previous frame
            feat_curr: Features from current frame
            feat_next: Features from next frame
        Returns:
            Temporally fused features
        """
        temporal_stack = torch.cat([feat_prev, feat_curr, feat_next], dim=1)
        return self.fusion(temporal_stack)


class ContextNetwork(nn.Module):
    def __init__(self, in_channels):
        super(ContextNetwork, self).__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(in_channels, 128, 1), nn.ReLU(True))
        self.layer1 = ResNetBlock(128, 128)
        self.layer2 = ResNetBlock(128, 128)
        self.layer3 = ResNetBlock(128, 128)
        self.conv_out = nn.Conv2d(128, 256, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.conv_out(x)


class UpdateBlock(nn.Module):
    def __init__(self, scale=1.0):
        super(UpdateBlock, self).__init__()
        self.motion_encoder = MotionEncoder()
        self.gru = ConvGRU(input_dim=128 + 32, hidden_dim=128)
        self.disp_head = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 1, 1)
        )
        self.scale = scale

    def forward(self, net, inp, corr, disp):
        motion_features = self.motion_encoder(disp, corr)
        inp_cat = torch.cat(
            [inp, F.interpolate(motion_features, size=inp.shape[-2:], mode='bilinear', align_corners=False)], 1)
        net = self.gru(net, inp_cat)
        return net, torch.tanh(self.disp_head(net)) * self.scale


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels, self.radius = num_levels, radius
        self.corr_pyramid = []
        corr = torch.einsum('bchw, bcij->bhwij', fmap1, fmap2)
        corr = corr.flatten(3).permute(0, 3, 1, 2).reshape(-1, 1, fmap2.shape[2], fmap2.shape[3])
        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            self.corr_pyramid.append(F.avg_pool2d(self.corr_pyramid[-1], 2, 2))

    def __call__(self, coords):
        r, B, _, H, W = self.radius, *coords.shape
        coords = coords.permute(0, 2, 3, 1)
        out = []
        for i, corr in enumerate(self.corr_pyramid):
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), -1)
            centroid = coords.reshape(B * H * W, 1, 1, 2) / 2 ** i
            coords_lvl = centroid + delta.view(1, -1, 1, 2)
            out.append(F.grid_sample(corr, coords_lvl, padding_mode='border', align_corners=True).view(B, H, W, -1))
        return torch.cat(out, -1).permute(0, 3, 1, 2).contiguous().float()


# --- Loss Function with Temporal Consistency ---
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.refl = nn.ReflectionPad2d(1)
        self.C1, self.C2 = 0.01 ** 2, 0.03 ** 2

    def forward(self, x, y):
        x, y = self.refl(x), self.refl(y)
        mu_x, mu_y = F.avg_pool2d(x, 3, 1), F.avg_pool2d(y, 3, 1)
        sig_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sig_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sig_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sig_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sig_x + sig_y + self.C2)
        return torch.clamp((1 - SSIM_n / (SSIM_d + 1e-8)) / 2, 0, 1)


class TemporalSelfSupervisedLoss(nn.Module):
    """Loss with temporal consistency constraint"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.photometric_weights = cfg.PHOTOMETRIC_LOSS_WEIGHTS
        self.ssim = SSIM()

    def forward(self, inputs, disp_preds_curr, disp_preds_prev=None, disp_preds_next=None):
        left_curr = inputs["left_curr"]
        right_curr = inputs["right_curr"]
        mask_curr = inputs["mask_curr"]

        photo_loss, smooth_loss, temporal_loss = 0.0, 0.0, 0.0

        # Standard photometric and smoothness loss for current frame
        for i, pred in enumerate(disp_preds_curr):
            i_w = self.cfg.DEEP_SUPERVISION_DECAY ** (len(disp_preds_curr) - i - 1)
            disp = F.interpolate(pred, size=left_curr.shape[-2:], mode='bilinear', align_corners=False) * (
                    left_curr.shape[2] / pred.shape[2])
            warped = self.inverse_warp(right_curr, disp)
            photo_loss += i_w * self.compute_photometric_loss(warped, left_curr, mask_curr)
            smooth_loss += i_w * self.compute_smoothness_loss(disp, left_curr)

        # Temporal consistency loss
        if disp_preds_prev is not None and disp_preds_next is not None:
            disp_curr = disp_preds_curr[-1]
            disp_prev = disp_preds_prev[-1]
            disp_next = disp_preds_next[-1]

            # Compute temporal smoothness: adjacent frames should have similar disparities
            temporal_loss = self.compute_temporal_consistency(disp_prev, disp_curr, disp_next, mask_curr)

        total_loss = photo_loss + self.cfg.SMOOTHNESS_WEIGHT * smooth_loss + \
                     self.cfg.TEMPORAL_WEIGHT * temporal_loss

        final_disp_up = F.interpolate(disp_preds_curr[-1], size=left_curr.shape[-2:],
                                      mode='bilinear', align_corners=False) * (
                                left_curr.shape[2] / disp_preds_curr[-1].shape[2])

        return {
            "total_loss": total_loss,
            "photometric_loss": photo_loss,
            "smoothness_loss": smooth_loss,
            "temporal_loss": temporal_loss,
            "warped_right_image": self.inverse_warp(right_curr, final_disp_up)
        }

    def compute_temporal_consistency(self, disp_prev, disp_curr, disp_next, mask):
        """Enforce temporal smoothness in disparity predictions"""
        # Upsample to match mask resolution
        h, w = mask.shape[-2:]
        disp_prev_up = F.interpolate(disp_prev, size=(h, w), mode='bilinear', align_corners=False)
        disp_curr_up = F.interpolate(disp_curr, size=(h, w), mode='bilinear', align_corners=False)
        disp_next_up = F.interpolate(disp_next, size=(h, w), mode='bilinear', align_corners=False)

        # Temporal differences (should be small for smooth motion)
        diff_prev_curr = torch.abs(disp_curr_up - disp_prev_up) * mask
        diff_curr_next = torch.abs(disp_next_up - disp_curr_up) * mask

        loss = (diff_prev_curr.sum() + diff_curr_next.sum()) / (mask.sum() * 2 + 1e-8)
        return loss

    def compute_photometric_loss(self, w, t, m):
        return self.photometric_weights[0] * (self.ssim(w, t) * m).sum() / (m.sum() + 1e-8) + \
            self.photometric_weights[1] * (torch.abs(w - t) * m).sum() / (m.sum() + 1e-8)

    def inverse_warp(self, features, disp):
        B, C, H, W = features.shape
        y, x = torch.meshgrid(torch.arange(H, device=features.device),
                              torch.arange(W, device=features.device), indexing='ij')
        grid = torch.stack([x, y], 0).float().repeat(B, 1, 1, 1)
        grid[:, 0] -= disp.squeeze(1)
        grid[:, 0] = 2 * grid[:, 0] / (W - 1) - 1
        grid[:, 1] = 2 * grid[:, 1] / (H - 1) - 1
        return F.grid_sample(features, grid.permute(0, 2, 3, 1), mode='bilinear',
                             padding_mode='border', align_corners=True)

    def compute_smoothness_loss(self, disp, img):
        disp_dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
        disp_dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]
        img_dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), 1, True)
        img_dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), 1, True)
        disp_dxx = disp_dx[:, :, :, 1:] - disp_dx[:, :, :, :-1]
        disp_dyy = disp_dy[:, :, 1:, :] - disp_dy[:, :, :-1, :]
        return (disp_dxx.abs() * torch.exp(-img_dx[:, :, :, :-1])).mean() + \
            (disp_dyy.abs() * torch.exp(-img_dy[:, :, :-1, :])).mean()


# --- Temporal Dataset ---
class TemporalWaveStereoDataset(Dataset):
    """Dataset that returns 3 consecutive frames"""

    def __init__(self, cfg: Config, is_validation=False):
        self.cfg, self.is_validation = cfg, is_validation
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"No images found in '{cfg.LEFT_IMAGE_DIR}'.")

        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            self.roi_left, self.roi_right = tuple(calib['roi_left']), tuple(calib['roi_right'])
        except Exception as e:
            sys.exit(f"Failed to load calibration file: {e}")

        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

        # Filter indices to ensure we can get prev and next frames
        self.valid_indices = [idx for idx in self.indices if 1 <= idx < num_frames - 1]

    def __len__(self):
        return len(self.valid_indices)

    def _load_and_process_frame(self, frame_idx, apply_augmentation=False):
        """Load and process a single frame (left + right)"""
        left_path = self.left_images[frame_idx]
        right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, 'right' + os.path.basename(left_path)[4:])

        left_raw = cv2.imread(left_path, 0)
        right_raw = cv2.imread(right_path, 0)

        if left_raw is None or right_raw is None:
            return None, None, None

        # Rectification
        left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

        # ROI cropping
        x, y, w, h = self.roi_left
        left_rect = left_rect[y:y + h, x:x + w]
        x, y, w, h = self.roi_right
        right_rect = right_rect[y:y + h, x:x + w]

        # Resize
        left_img = cv2.resize(left_rect, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
        right_img = cv2.resize(right_rect, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))

        # Augmentation (only for training, same for all frames in sequence)
        if apply_augmentation:
            if np.random.rand() < 0.5:
                brightness = np.random.uniform(0.7, 1.3)
                left_img = np.clip(left_img * brightness, 0, 255).astype(np.uint8)
                right_img = np.clip(right_img * brightness, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.5:
                contrast = np.random.uniform(0.7, 1.3)
                left_img = np.clip((left_img - left_img.mean()) * contrast + left_img.mean(), 0, 255).astype(np.uint8)
                right_img = np.clip((right_img - right_img.mean()) * contrast + right_img.mean(), 0, 255).astype(
                    np.uint8)

        _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

        def to_tensor(img):
            return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0

        return to_tensor(left_img), to_tensor(right_img), torch.from_numpy(mask).float().unsqueeze(0) / 255.0

    def __getitem__(self, idx):
        try:
            curr_idx = self.valid_indices[idx]
            prev_idx = curr_idx - 1
            next_idx = curr_idx + 1

            # Load 3 consecutive frames
            apply_aug = not self.is_validation and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY

            left_prev, right_prev, mask_prev = self._load_and_process_frame(prev_idx, apply_aug)
            left_curr, right_curr, mask_curr = self._load_and_process_frame(curr_idx, apply_aug)
            left_next, right_next, mask_next = self._load_and_process_frame(next_idx, apply_aug)

            if any(x is None for x in [left_prev, left_curr, left_next]):
                return None

            return {
                'left_prev': left_prev, 'right_prev': right_prev, 'mask_prev': mask_prev,
                'left_curr': left_curr, 'right_curr': right_curr, 'mask_curr': mask_curr,
                'left_next': left_next, 'right_next': right_next, 'mask_next': mask_next
            }
        except Exception as e:
            print(f"Warning at idx {idx}: {e}")
            return None


# --- Temporal RAFT Model ---
class TemporalDINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.dino = self._load_dino_model()

        for p in self.dino.parameters():
            p.requires_grad = False

        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

        # Temporal fusion module
        self.temporal_fusion = TemporalFusion(self.feature_dim)

        self.context_net = ContextNetwork(self.feature_dim)
        self.update_block = UpdateBlock(scale=self.cfg.DISP_UPDATE_SCALE)
        print(
            f"Temporal RAFT model: {self.cfg.ITERATIONS} iterations, {self.cfg.TEMPORAL_FRAMES} frames, DINOv3 FROZEN.")

    def _load_dino_model(self):
        try:
            return AutoModel.from_pretrained(self.cfg.DINO_LOCAL_PATH, local_files_only=True)
        except Exception as e:
            print(f"[FATAL] loading DINOv3: {e}")
            sys.exit(1)

    def get_features(self, image):
        with torch.no_grad():
            features = self.dino(image).last_hidden_state
        b, _, h, w = image.shape
        start_idx = 1 + self.num_register_tokens
        patch_tokens = features[:, start_idx:, :]
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        return patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, h_feat, w_feat)

    def initialize_flow(self, image):
        b, _, h, w = image.shape
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        return torch.zeros(b, 2, h_feat, w_feat, device=image.device)

    def get_grid(self, b, h, w, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        return torch.stack([x, y], 0).float().unsqueeze(0).repeat(b, 1, 1, 1)

    def forward(self, left_prev, left_curr, left_next, right_curr):
        """
        Args:
            left_prev: Previous left frame
            left_curr: Current left frame
            left_next: Next left frame
            right_curr: Current right frame (stereo pair)
        """
        # Extract features from all frames
        feat_left_prev = self.get_features(left_prev)
        feat_left_curr = self.get_features(left_curr)
        feat_left_next = self.get_features(left_next)
        feat_right_curr = self.get_features(right_curr)

        # Temporally fuse left features
        feat_left_fused = self.temporal_fusion(feat_left_prev, feat_left_curr, feat_left_next)

        # Normalize
        fmap1 = F.normalize(feat_left_fused, 2, 1)
        fmap2 = F.normalize(feat_right_curr, 2, 1)# Build correlation volume
        corr_fn = CorrBlock(fmap1.float(), fmap2.float())

        # Context network on fused features
        cnet = self.context_net(fmap1)
        net, inp = torch.split(cnet, [128, 128], 1)
        net, inp = torch.tanh(net), torch.relu(inp)

        # Iterative refinement
        flow = self.initialize_flow(left_curr)
        b, c, h_feat, w_feat = fmap1.shape
        coords0 = self.get_grid(b, h_feat, w_feat, fmap1.device)
        preds = []

        for _ in range(self.cfg.ITERATIONS):
            flow = flow.detach()
            coords1 = coords0 - flow
            corr = corr_fn(coords1)
            net, delta_flow = self.update_block(net, inp, corr, flow)
            flow = flow + delta_flow
            preds.append(flow)

        return preds


# --- Evaluation and Trainer ---
class EvaluationMetrics:
    @staticmethod
    def compute_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    @staticmethod
    def evaluate_reconstruction(inputs, outputs, loss_components):
        left_img = inputs["left_curr"]
        if "warped_right_image" not in loss_components or loss_components["warped_right_image"] is None:
            return {"psnr": 0.0}
        return {"psnr": EvaluationMetrics.compute_psnr(left_img, loss_components["warped_right_image"])}


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR, self.timestamp)
        for d in ["checkpoints", "visualizations", "logs", "tensorboard"]:
            os.makedirs(os.path.join(self.run_dir, d), exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        train_ds = TemporalWaveStereoDataset(cfg, False)
        val_ds = TemporalWaveStereoDataset(cfg, True)
        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_fn, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=0, pin_memory=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tensorboard")) if SummaryWriter else None
        self.model = TemporalDINOv3StereoModel(cfg).to(self.device)
        self.loss_fn = TemporalSelfSupervisedLoss(cfg)

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=cfg.LEARNING_RATE, weight_decay=1e-4)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-7
        )

        self.evaluator = EvaluationMetrics()
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION)
        self.step = 0
        self.log_file = os.path.join(self.run_dir, "logs", "training_log.json")
        self.loss_keys = ['total', 'photometric', 'smoothness', 'temporal']
        self.metric_keys = ['psnr']
        self.history = {
            'train': {k: [] for k in self.loss_keys + self.metric_keys},
            'val': {k: [] for k in self.loss_keys + self.metric_keys}
        }

    def train(self):
        print("\n--- Starting Temporal Enhanced Training ---")
        print(f"Using {self.cfg.TEMPORAL_FRAMES} frames with temporal weight {self.cfg.TEMPORAL_WEIGHT}")
        best_val_psnr, epochs_no_improve = 0.0, 0

        for epoch in range(self.cfg.NUM_EPOCHS):
            train_results = self._run_epoch(epoch, True)
            self._log_epoch_results('train', epoch, train_results)

            with torch.no_grad():
                val_results = self._run_epoch(epoch, False)
            self._log_epoch_results('val', epoch, val_results)

            avg_val_psnr = val_results.get('psnr', 0.0)

            print(
                f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> "
                f"Train Loss: {train_results.get('total', 0.0):.4f} | "
                f"Val PSNR: {avg_val_psnr:.2f} dB | "
                f"Temporal Loss: {train_results.get('temporal', 0.0):.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            self.update_log_file(epoch)
            if self.cfg.VISUALIZE_TRAINING:
                self.plot_training_history()

            if avg_val_psnr > best_val_psnr:
                best_val_psnr, epochs_no_improve = avg_val_psnr, 0
                torch.save(self.model.state_dict(),
                          os.path.join(self.run_dir, "checkpoints", "best_model_temporal.pth"))
                print(f"  Val PSNR improved to {best_val_psnr:.2f} dB. Model saved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.EARLY_STOPPING_PATIENCE:
                    print(f"--- Early stopping after {epoch + 1} epochs. ---")
                    break

            self.scheduler.step()

        print("--- Training complete! ---")
        if self.writer:
            self.writer.close()

    def _pad_inputs(self, *tensors):
        """Pad all input tensors to match patch size"""
        _, _, h, w = tensors[0].shape
        pad_h = (self.model.patch_size - h % self.model.patch_size) % self.model.patch_size
        pad_w = (self.model.patch_size - w % self.model.patch_size) % self.model.patch_size
        if pad_h > 0 or pad_w > 0:
            return [F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0) for x in tensors]
        return list(tensors)

    def _run_epoch(self, epoch, is_training):
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [{'Train' if is_training else 'Val'}]")
        epoch_results = {k: 0.0 for k in self.loss_keys + self.metric_keys}

        for data in pbar:
            if data is None:
                continue

            # Move all data to device
            data = {k: v.to(self.device) for k, v in data.items()}

            # Pad inputs
            left_prev, left_curr, left_next, right_prev, right_curr, right_next = self._pad_inputs(
                data['left_prev'], data['left_curr'], data['left_next'],
                data['right_prev'], data['right_curr'], data['right_next']
            )
            mask_prev, mask_curr, mask_next = self._pad_inputs(
                data['mask_prev'], data['mask_curr'], data['mask_next']
            )

            with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION):
                # Forward pass for all 3 frames
                flow_preds_prev = self.model(left_prev, left_prev, left_curr, right_prev)
                flow_preds_curr = self.model(left_prev, left_curr, left_next, right_curr)
                flow_preds_next = self.model(left_curr, left_next, left_next, right_next)

                # Extract disparity (x-component of flow)
                disp_preds_prev = [flow[:, 0:1, :, :] for flow in flow_preds_prev]
                disp_preds_curr = [flow[:, 0:1, :, :] for flow in flow_preds_curr]
                disp_preds_next = [flow[:, 0:1, :, :] for flow in flow_preds_next]

                # Compute loss with temporal consistency
                loss_comps = self.loss_fn(
                    {
                        "left_curr": left_curr,
                        "right_curr": right_curr,
                        "mask_curr": mask_curr
                    },
                    disp_preds_curr,
                    disp_preds_prev,
                    disp_preds_next
                )
                loss = loss_comps["total_loss"]

            if is_training:
                if torch.isfinite(loss):
                    accum_steps = self.cfg.GRADIENT_ACCUMULATION_STEPS
                    self.scaler.scale(loss / accum_steps).backward()
                    if (self.step + 1) % accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    print(f"\nWarning: Invalid loss at step {self.step}. Skipping.")

            final_disp = disp_preds_curr[-1]
            metrics = self.evaluator.evaluate_reconstruction(
                {"left_curr": left_curr},
                {"disparity": final_disp},
                loss_comps
            )

            for k in self.loss_keys:
                if f"{k}_loss" in loss_comps:
                    epoch_results[k] += loss_comps[f"{k}_loss"].item()
            for k in self.metric_keys:
                if k in metrics:
                    epoch_results[k] += metrics[k]

            pbar.set_postfix({
                'loss': loss.item(),
                'psnr': metrics.get('psnr', 0.0),
                'temp': loss_comps.get('temporal_loss', 0.0).item() if isinstance(loss_comps.get('temporal_loss', 0.0), torch.Tensor) else loss_comps.get('temporal_loss', 0.0)
            })

            if is_training:
                if self.writer:
                    self.writer.add_scalar('Loss/step_train', loss.item(), self.step)
                    if loss_comps.get('temporal_loss', 0.0) > 0:
                        temp_loss_val = loss_comps['temporal_loss'].item() if isinstance(loss_comps['temporal_loss'], torch.Tensor) else loss_comps['temporal_loss']
                        self.writer.add_scalar('Loss/temporal', temp_loss_val, self.step)

                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(left_curr, mask_curr, final_disp, loss_comps, self.step, "train")

                self.step += 1

        num_batches = len(loader)
        return {k: v / num_batches for k, v in epoch_results.items()} if num_batches > 0 else epoch_results

    def _log_epoch_results(self, phase, epoch, results):
        for k, v in results.items():
            self.history[phase][k].append(v)
            if self.writer:
                metric_type = 'Loss' if k in self.loss_keys else 'Metrics'
                self.writer.add_scalar(f"{metric_type}/{phase}_{k}", v, epoch)

        if self.writer and phase == 'val' and self.history['train']['total']:
            self.writer.add_scalars('Loss/epoch_comparison',
                                    {'train': self.history['train']['total'][-1],
                                     'val': results.get('total', 0)}, epoch)
            if 'psnr' in self.history['train'] and self.history['train']['psnr']:
                self.writer.add_scalars('PSNR/epoch_comparison',
                                        {'train': self.history['train']['psnr'][-1],
                                         'val': results.get('psnr', 0)}, epoch)

    def visualize(self, left_tensor, mask_tensor, final_disp, loss_components, step, phase):
        left_img = (left_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = mask_tensor[0, 0].cpu().numpy()
        pred_disp_up = F.interpolate(final_disp, size=left_img.shape[:2], mode='bilinear', align_corners=False)
        pred_disp = pred_disp_up[0, 0].cpu().detach().numpy()
        masked_disp = np.ma.masked_where(mask == 0, pred_disp)
        warped_right = (loss_components["warped_right_image"][0].cpu().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Temporal RAFT Model - Step: {step} ({phase})', fontsize=16)

        axes[0, 0].imshow(left_img)
        axes[0, 0].set_title("Input Left Image (Current Frame)")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(warped_right)
        axes[0, 1].set_title("Reconstructed Left from Right")
        axes[0, 1].axis('off')

        im = axes[1, 0].imshow(masked_disp, cmap='viridis')
        axes[1, 0].set_title("Predicted Disparity (Masked)")
        axes[1, 0].set_facecolor('black')
        axes[1, 0].axis('off')
        fig.colorbar(im, ax=axes[1, 0])

        diff_map = np.abs(left_img.astype(float) - warped_right.astype(float)).mean(2)
        im_diff = axes[1, 1].imshow(diff_map, cmap='hot')
        axes[1, 1].set_title("Photometric Error")
        axes[1, 1].axis('off')
        fig.colorbar(im_diff, ax=axes[1, 1])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.run_dir, "visualizations", f"{phase}_step_{step:06d}.png")
        plt.savefig(save_path)
        if self.writer:
            self.writer.add_figure(f'Visualization/{phase}', fig, step)
        plt.close(fig)

    def plot_training_history(self):
        if not self.history['train']['total']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal RAFT Model Training History', fontsize=16)

        # Total Loss
        axes[0, 0].plot(self.history['train']['total'], label='Train Loss')
        axes[0, 0].plot(self.history['val']['total'], label='Val Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Loss")

        # PSNR
        axes[0, 1].plot(self.history['train']['psnr'], label='Train PSNR')
        axes[0, 1].plot(self.history['val']['psnr'], label='Val PSNR')
        axes[0, 1].set_title('PSNR')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_xlabel("Epochs")
        axes[0, 1].set_ylabel("PSNR (dB)")

        # Temporal Loss
        axes[1, 0].plot(self.history['train']['temporal'], label='Train Temporal Loss')
        axes[1, 0].plot(self.history['val']['temporal'], label='Val Temporal Loss')
        axes[1, 0].set_title('Temporal Consistency Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Loss")

        # Loss components
        axes[1, 1].plot(self.history['train']['photometric'], label='Photometric')
        axes[1, 1].plot(self.history['train']['smoothness'], label='Smoothness')
        axes[1, 1].plot(self.history['train']['temporal'], label='Temporal')
        axes[1, 1].set_title('Loss Components (Train)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].set_ylabel("Loss")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.run_dir, "visualizations", "training_history.png")
        plt.savefig(save_path)
        plt.close(fig)

    def update_log_file(self, epoch):
        log_data = {'config': asdict(self.cfg), 'epoch': epoch, 'history': self.history}
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)


if __name__ == "__main__":
    cfg = Config()
    try:
        auto_tune_config(cfg)
    except Exception as e:
        print(f"[WARNING] Auto-tuning failed: {e}. Using default config.")

    if cfg.BATCH_SIZE == 0:
        print("[WARNING] Auto-tuned batch size is 0. Setting to 1.")
        cfg.BATCH_SIZE = 1

    trainer = Trainer(cfg)
    trainer.train()