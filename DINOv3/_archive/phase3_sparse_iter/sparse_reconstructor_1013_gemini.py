# sparse_matching_stereo.py
# Sparse keypoint matching approach for sparse specular water surface reconstruction
# Key idea: Only match high-intensity specular points, ignore 95% black background

import os
import sys
import glob
import time
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
    print("=" * 80 + "\n[FATAL ERROR]: transformers not found\n" + "=" * 80)
    sys.exit(1)

PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


@dataclass
class Config:
    """Configuration for sparse matching"""
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")
    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs_sparse")
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")

    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 100
    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 512
    MASK_THRESHOLD: int = 30

    # Sparse matching settings
    MAX_KEYPOINTS: int = 512  # Maximum keypoints per image
    NMS_RADIUS: int = 3  # Non-maximum suppression radius
    PATCH_SIZE: int = 5  # Patch size for photometric loss (5x5)

    # Matching network settings
    FEATURE_DIM: int = 768  # DINOv3 feature dimension
    NUM_ATTENTION_LAYERS: int = 4  # Number of attention layers
    NUM_HEADS: int = 8  # Number of attention heads
    DISPARITY_CONSTRAINT_Y_THRESHOLD: int = 2  # Y-pixel tolerance for matching

    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    GRADIENT_ACCUMULATION_STEPS: int = 1

    USE_MIXED_PRECISION: bool = True
    PHOTOMETRIC_WEIGHT: float = 1.0
    SMOOTHNESS_WEIGHT: float = 0.1  # Increased from 0.01 for more regularization

    USE_ADVANCED_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.8

    EARLY_STOPPING_PATIENCE: int = 25


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
        print("[WARNING] Could not probe GPU. Using default settings.")
        return cfg

    print(f"Available GPU Memory: {free_mem_mb} MB")

    # Sparse matching needs less memory than dense methods
    if free_mem_mb < 8000:
        cfg.BATCH_SIZE, cfg.MAX_KEYPOINTS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 256, 4, 0.7
    elif free_mem_mb < 12000:
        cfg.BATCH_SIZE, cfg.MAX_KEYPOINTS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 512, 2, 0.8
    else:
        # Cap batch size to 1 to reduce instantaneous hardware stress, use accumulation for larger effective batch
        cfg.BATCH_SIZE, cfg.MAX_KEYPOINTS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 512, 2, 1.0

    cfg.IMAGE_WIDTH = int((cfg.IMAGE_WIDTH * scale) // patch_size) * patch_size
    cfg.IMAGE_HEIGHT = int((cfg.IMAGE_HEIGHT * scale) // patch_size) * patch_size

    print(f"  Resolution: {cfg.IMAGE_WIDTH}x{cfg.IMAGE_HEIGHT}, Batch: {cfg.BATCH_SIZE}, "
          f"Max Keypoints: {cfg.MAX_KEYPOINTS}")
    return cfg


# --- 1. Sparse Keypoint Detector ---
class SparseKeypointDetector(nn.Module):
    """Detect sparse keypoints (bright specular points) on grayscale images"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_keypoints = cfg.MAX_KEYPOINTS
        self.nms_radius = cfg.NMS_RADIUS

    def forward(self, gray_image, mask):
        """
        Args:
            gray_image: (B, 1, H, W) - grayscale image (0-1 range)
            mask: (B, 1, H, W) - valid region mask
        Returns:
            keypoints: (B, N, 2) - (x, y) coordinates, N <= max_keypoints
            scores: (B, N) - intensity scores
        """
        B, _, H, W = gray_image.shape
        device = gray_image.device

        keypoints_list = []
        scores_list = []

        for b in range(B):
            img = gray_image[b, 0]  # (H, W)
            m = mask[b, 0]  # (H, W)

            # Apply mask
            img_masked = img * m

            # Find top-K brightest points
            img_flat = img_masked.view(-1)
            values, indices = torch.topk(img_flat, min(self.max_keypoints * 4, img_flat.numel()))

            # Convert to 2D coordinates
            y_coords = indices // W
            x_coords = indices % W

            # Filter out zero-valued points (background)
            valid_mask = values > 0.1
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            values = values[valid_mask]

            if len(x_coords) == 0:
                # No keypoints found, return dummy
                keypoints_list.append(torch.zeros(1, 2, device=device))
                scores_list.append(torch.zeros(1, device=device))
                continue

            # Non-maximum suppression
            kp = torch.stack([x_coords, y_coords], dim=1).float()  # (N, 2)
            kp_nms, scores_nms = self.non_maximum_suppression(kp, values)

            # Limit to max_keypoints
            if len(kp_nms) > self.max_keypoints:
                kp_nms = kp_nms[:self.max_keypoints]
                scores_nms = scores_nms[:self.max_keypoints]

            keypoints_list.append(kp_nms)
            scores_list.append(scores_nms)

        # Pad to same length
        max_len = max(len(kp) for kp in keypoints_list)

        keypoints_padded = []
        scores_padded = []

        for kp, sc in zip(keypoints_list, scores_list):
            pad_len = max_len - len(kp)
            if pad_len > 0:
                kp_pad = torch.cat([kp, torch.zeros(pad_len, 2, device=device)], dim=0)
                sc_pad = torch.cat([sc, torch.zeros(pad_len, device=device)], dim=0)
            else:
                kp_pad, sc_pad = kp, sc
            keypoints_padded.append(kp_pad)
            scores_padded.append(sc_pad)

        keypoints = torch.stack(keypoints_padded, dim=0)  # (B, N, 2)
        scores = torch.stack(scores_padded, dim=0)  # (B, N)

        return keypoints, scores

    def non_maximum_suppression(self, keypoints, scores):
        """
        Simple NMS to remove clustered points.
        This operation is performed on the CPU for stability.
        """
        if len(keypoints) == 0:
            return keypoints, scores

        original_device = keypoints.device

        # Move to CPU for stable processing
        kp_cpu = keypoints.cpu()
        scores_cpu = scores.cpu()

        # Sort by score descending
        sorted_indices = torch.argsort(scores_cpu, descending=True)
        kp_sorted = kp_cpu[sorted_indices]
        scores_sorted = scores_cpu[sorted_indices]

        keep_mask = torch.ones(len(kp_sorted), dtype=torch.bool)

        for i in range(len(kp_sorted)):
            if not keep_mask[i]:
                continue

            # Compute distance to all other points
            dist = torch.norm(kp_sorted[i:i + 1] - kp_sorted[i + 1:], dim=1)

            # Suppress nearby points
            suppress_mask = dist < self.nms_radius

            # Find indices to suppress and update the mask
            # This is a safer way than sliced assignment
            indices_to_check = torch.arange(i + 1, len(kp_sorted))
            indices_to_suppress = indices_to_check[suppress_mask]
            keep_mask[indices_to_suppress] = False

        # Move the final results back to the original device
        final_kp = kp_sorted[keep_mask].to(original_device)
        final_scores = scores_sorted[keep_mask].to(original_device)

        return final_kp, final_scores


# --- 2. DINOv3 Feature Extractor (keeping your original approach) ---
class DINOv3FeatureExtractor(nn.Module):
    """Extract DINOv3 features at keypoint locations"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.dino = self._load_dino_model()

        for p in self.dino.parameters():
            p.requires_grad = False

        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

    def _load_dino_model(self):
        try:
            return AutoModel.from_pretrained(self.cfg.DINO_LOCAL_PATH, local_files_only=True)
        except Exception as e:
            print(f"[FATAL] loading DINOv3: {e}")
            sys.exit(1)

    def get_feature_map(self, image):
        """
        Extract spatial feature map from DINOv3
        Args:
            image: (B, 3, H, W) - RGB image
        Returns:
            feat_map: (B, 768, H//16, W//16)
        """
        with torch.no_grad():
            features = self.dino(image).last_hidden_state

        b, _, h, w = image.shape
        start_idx = 1 + self.num_register_tokens
        patch_tokens = features[:, start_idx:, :]
        h_feat, w_feat = h // self.patch_size, w // self.patch_size

        feat_map = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, h_feat, w_feat)
        return feat_map

    def forward(self, image, keypoints):
        """
        Sample features at keypoint locations
        Args:
            image: (B, 3, H, W)
            keypoints: (B, N, 2) - (x, y) coordinates
        Returns:
            descriptors: (B, N, 768)
        """
        B, N, _ = keypoints.shape
        feat_map = self.get_feature_map(image)  # (B, 768, H//16, W//16)

        _, C, H_feat, W_feat = feat_map.shape
        _, _, H_img, W_img = image.shape

        # Normalize keypoint coordinates to feature map space
        grid = keypoints.clone()
        grid[:, :, 0] = 2 * (grid[:, :, 0] / W_img) - 1  # x to [-1, 1]
        grid[:, :, 1] = 2 * (grid[:, :, 1] / H_img) - 1  # y to [-1, 1]
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)

        # Sample features using grid_sample
        descriptors = F.grid_sample(
            feat_map, grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        ).squeeze(3).permute(0, 2, 1)  # (B, N, 768)

        return descriptors


# --- 3. Attention-based Sparse Matching Network ---
class PositionalEncoding(nn.Module):
    """2D positional encoding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(2, dim)

    def forward(self, positions, image_size):
        """
        Args:
            positions: (B, N, 2) - (x, y) coordinates
            image_size: (H, W)
        Returns:
            encoding: (B, N, dim)
        """
        H, W = image_size
        pos_normalized = positions.clone()
        pos_normalized[:, :, 0] = pos_normalized[:, :, 0] / W
        pos_normalized[:, :, 1] = pos_normalized[:, :, 1] / H
        return self.proj(pos_normalized)


class SelfAttentionLayer(nn.Module):
    """Self-attention with positional encoding"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, features, pos_enc):
        """
        Args:
            features: (B, N, dim)
            pos_enc: (B, N, dim)
        Returns:
            features: (B, N, dim)
        """
        # Self-attention with residual
        features_pos = features + pos_enc
        attn_out, _ = self.attn(features_pos, features_pos, features_pos)
        features = self.norm1(features + attn_out)

        # FFN with residual
        ffn_out = self.ffn(features)
        features = self.norm2(features + ffn_out)

        return features


class CrossAttentionLayer(nn.Module):
    """Cross-attention between left and right features"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_query, feat_kv):
        """
        Args:
            feat_query: (B, N_q, dim) - query features (left)
            feat_kv: (B, N_kv, dim) - key/value features (right)
        Returns:
            features: (B, N_q, dim)
        """
        attn_out, attn_weights = self.attn(feat_query, feat_kv, feat_kv)
        return self.norm(feat_query + attn_out), attn_weights


class SparseMatchingNetwork(nn.Module):
    """Attention-based sparse feature matching"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim = cfg.FEATURE_DIM
        num_layers = cfg.NUM_ATTENTION_LAYERS
        num_heads = cfg.NUM_HEADS

        # Positional encoding
        self.pos_enc = PositionalEncoding(dim)

        # Self-attention layers
        self.self_attn_left = nn.ModuleList([
            SelfAttentionLayer(dim, num_heads) for _ in range(num_layers)
        ])
        self.self_attn_right = nn.ModuleList([
            SelfAttentionLayer(dim, num_heads) for _ in range(num_layers)
        ])

        # Cross-attention layer
        self.cross_attn = CrossAttentionLayer(dim, num_heads)

    def forward(self, desc_left, desc_right, kp_left, kp_right, image_size):
        """
        Args:
            desc_left: (B, N_l, 768)
            desc_right: (B, N_r, 768)
            kp_left: (B, N_l, 2)
            kp_right: (B, N_r, 2)
            image_size: (H, W)
        Returns:
            match_scores: (B, N_l, N_r) - matching scores
            disparity: (B, N_l) - predicted disparity for each left keypoint
        """
        # Positional encoding
        pos_left = self.pos_enc(kp_left, image_size)
        pos_right = self.pos_enc(kp_right, image_size)

        # Self-attention to enhance features
        feat_left = desc_left
        feat_right = desc_right

        for self_l, self_r in zip(self.self_attn_left, self.self_attn_right):
            feat_left = self_l(feat_left, pos_left)
            feat_right = self_r(feat_right, pos_right)

        # Cross-attention: left queries right
        feat_left_enhanced, attn_weights = self.cross_attn(feat_left, feat_right)

        # Compute matching scores (cosine similarity)
        feat_left_norm = F.normalize(feat_left_enhanced, dim=2)
        feat_right_norm = F.normalize(feat_right, dim=2)
        match_scores = torch.bmm(feat_left_norm, feat_right_norm.transpose(1, 2))  # (B, N_l, N_r)

        # Enforce epipolar constraints and non-negative disparity before softmax
        x_left = kp_left[:, :, 0].unsqueeze(2)  # (B, N_l, 1)
        y_left = kp_left[:, :, 1].unsqueeze(2)  # (B, N_l, 1)
        x_right = kp_right[:, :, 0].unsqueeze(1)  # (B, 1, N_r)
        y_right = kp_right[:, :, 1].unsqueeze(1)  # (B, 1, N_r)

        # 1. Disparity must be non-negative: x_left >= x_right
        # 2. Y-coordinates must be aligned: abs(y_left - y_right) < threshold
        valid_x_mask = (x_left >= x_right)
        valid_y_mask = (y_left - y_right).abs() < self.cfg.DISPARITY_CONSTRAINT_Y_THRESHOLD
        constraint_mask = (valid_x_mask & valid_y_mask)

        # Apply mask to scores, filling invalid matches with a large negative number
        # that is safe for the current dtype (e.g., float16)
        match_scores = match_scores.masked_fill(~constraint_mask, torch.finfo(match_scores.dtype).min)

        # Predict disparity using soft matching
        match_probs = F.softmax(match_scores * 10, dim=2)  # (B, N_l, N_r)

        # Disparity = x_left - weighted_x_right
        # Note: We use the expanded x_left and x_right from before the unsqueeze
        disparity_matrix = kp_left[:, :, 0].unsqueeze(2) - kp_right[:, :, 0].unsqueeze(1)  # (B, N_l, N_r)
        disparity_pred = (disparity_matrix * match_probs).sum(dim=2)  # (B, N_l)

        # Replace any NaNs that might occur if a keypoint has no valid matches
        disparity_pred = torch.nan_to_num(disparity_pred)

        return match_scores, disparity_pred


# --- 4. Sparse Photometric Loss ---
class SparsePhotometricLoss(nn.Module):
    """Photometric loss on sparse keypoint patches, computed in a vectorized manner."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.PATCH_SIZE
        self.hp = cfg.PATCH_SIZE // 2

        # Create patch offsets once, requires no device at init
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-self.hp, self.hp + 1),
            torch.arange(-self.hp, self.hp + 1),
            indexing='ij'
        )
        # Shape: (1, 1, patch_size**2, 2)
        self.offsets = torch.stack([offsets_x, offsets_y], dim=-1).view(1, 1, self.patch_size ** 2, 2)

    def forward(self, left_img, right_img, keypoints_left, disparity, scores_left):
        """Vectorized calculation of photometric loss."""
        B, _, H, W = left_img.shape
        N = keypoints_left.shape[1]
        device = left_img.device

        # Ensure offsets are on the correct device
        offsets = self.offsets.to(device)

        # 1. Create a mask for valid (non-padded) keypoints
        valid_kp_mask = (scores_left > 0.1).unsqueeze(-1)  # (B, N, 1)

        # 2. Calculate coordinates for right image patches
        coords_right = keypoints_left.clone()
        coords_right[..., 0] -= disparity  # Corrected line

        # 3. Generate pixel coordinates for all patches
        patch_coords_left = keypoints_left.unsqueeze(2) + offsets  # (B, N, 25, 2)
        patch_coords_right = coords_right.unsqueeze(2) + offsets  # (B, N, 25, 2)

        # 4. Create mask for patches fully within image bounds
        min_coords_l = patch_coords_left.min(dim=2, keepdim=True)[0]
        max_coords_l = patch_coords_left.max(dim=2, keepdim=True)[0]
        min_coords_r = patch_coords_right.min(dim=2, keepdim=True)[0]
        max_coords_r = patch_coords_right.max(dim=2, keepdim=True)[0]

        bounds_mask_l = (min_coords_l[..., 0] >= 0) & (min_coords_l[..., 1] >= 0) & \
                        (max_coords_l[..., 0] < W) & (max_coords_l[..., 1] < H)
        bounds_mask_r = (min_coords_r[..., 0] >= 0) & (min_coords_r[..., 1] >= 0) & \
                        (max_coords_r[..., 0] < W) & (max_coords_r[..., 1] < H)

        final_mask = valid_kp_mask & bounds_mask_l & bounds_mask_r  # (B, N, 1)
        num_valid = torch.sum(final_mask)

        if num_valid == 0:
            photo_loss = torch.tensor(0.0, device=device)
            smooth_loss = self._compute_sparse_smoothness(keypoints_left, disparity, scores_left)
            return photo_loss, smooth_loss

        # 5. Normalize coordinates for grid_sample: from [0, W-1] to [-1, 1]
        patch_coords_left[..., 0] = 2 * patch_coords_left[..., 0] / (W - 1) - 1
        patch_coords_left[..., 1] = 2 * patch_coords_left[..., 1] / (H - 1) - 1
        patch_coords_right[..., 0] = 2 * patch_coords_right[..., 0] / (W - 1) - 1
        patch_coords_right[..., 1] = 2 * patch_coords_right[..., 1] / (H - 1) - 1

        # 6. Sample patches using grid_sample
        # Input grid shape: (B, H_out, W_out, 2), here we use (B, N, 25, 2)
        # Output shape: (B, C, H_out, W_out), here (B, 1, N, 25)
        patches_left = F.grid_sample(left_img, patch_coords_left, mode='bilinear', align_corners=True,
                                     padding_mode='border')
        patches_right = F.grid_sample(right_img, patch_coords_right, mode='bilinear', align_corners=True,
                                      padding_mode='border')

        # 7. Compute L1 loss
        # Reshape patches from (B, 1, N, 25) to (B, N, 25)
        loss_per_patch = F.l1_loss(patches_left.squeeze(1), patches_right.squeeze(1), reduction='none').mean(
            dim=2)  # (B, N)

        # 8. Apply mask and compute final loss
        masked_loss = loss_per_patch * final_mask.squeeze(-1)
        photo_loss = masked_loss.sum() / num_valid

        # Smoothness loss (unchanged)
        smooth_loss = self._compute_sparse_smoothness(keypoints_left, disparity, scores_left)

        return photo_loss, smooth_loss

    def _compute_sparse_smoothness(self, keypoints, disparity, scores):
        """Smoothness: nearby keypoints should have similar disparity"""
        B, N, _ = keypoints.shape
        smooth_loss = 0.0
        valid_pairs = 0

        for b in range(B):
            kp = keypoints[b]  # (N, 2)
            disp = disparity[b]  # (N,)
            sc = scores[b]  # (N,)

            # Only consider valid keypoints
            valid_mask = sc > 0.1
            kp_valid = kp[valid_mask]
            disp_valid = disp[valid_mask]

            if len(kp_valid) < 2:
                continue

            # Compute pairwise distances
            dist = torch.cdist(kp_valid, kp_valid)  # (M, M)

            # Only penalize nearby pairs (within 20 pixels)
            neighbor_mask = (dist < 20) & (dist > 0)

            disp_diff = (disp_valid.unsqueeze(0) - disp_valid.unsqueeze(1)).abs()
            smooth_loss += (disp_diff * neighbor_mask.float()).sum()
            valid_pairs += neighbor_mask.sum()

        if valid_pairs > 0:
            smooth_loss /= valid_pairs

        return smooth_loss


# --- 5. Dataset (same as before) ---
class RectifiedWaveStereoDataset(Dataset):
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        try:
            left_path = self.left_images[self.indices[idx]]
            right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, 'right' + os.path.basename(left_path)[4:])
            left_raw, right_raw = cv2.imread(left_path, 0), cv2.imread(right_path, 0)
            if left_raw is None or right_raw is None:
                return None

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

            # Augmentation
            if not self.is_validation and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
                if np.random.rand() < 0.5:
                    brightness = np.random.uniform(0.7, 1.3)
                    left_img = np.clip(left_img * brightness, 0, 255).astype(np.uint8)
                    right_img = np.clip(right_img * brightness, 0, 255).astype(np.uint8)
                if np.random.rand() < 0.5:
                    contrast = np.random.uniform(0.7, 1.3)
                    left_img = np.clip((left_img - left_img.mean()) * contrast + left_img.mean(), 0, 255).astype(
                        np.uint8)
                    right_img = np.clip((right_img - right_img.mean()) * contrast + right_img.mean(), 0, 255).astype(
                        np.uint8)

            # Mask
            _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Convert to tensors
            left_gray = torch.from_numpy(left_img).float().unsqueeze(0) / 255.0  # (1, H, W)
            right_gray = torch.from_numpy(right_img).float().unsqueeze(0) / 255.0

            # For DINOv3, need RGB (convert gray to 3-channel)
            left_rgb = torch.from_numpy(cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0
            right_rgb = torch.from_numpy(cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0

            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

            return {
                'left_gray': left_gray,
                'right_gray': right_gray,
                'left_rgb': left_rgb,
                'right_rgb': right_rgb,
                'mask': mask_tensor
            }
        except Exception as e:
            print(f"Warning at idx {idx}: {e}")
            return None


# --- 6. Complete Model ---
class SparseMatchingStereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.keypoint_detector = SparseKeypointDetector(cfg)
        self.feature_extractor = DINOv3FeatureExtractor(cfg)
        self.matcher = SparseMatchingNetwork(cfg)

        print(f"Sparse Matching Model: max {cfg.MAX_KEYPOINTS} keypoints, "
              f"{cfg.NUM_ATTENTION_LAYERS} attention layers")

    def forward(self, left_gray, right_gray, left_rgb, right_rgb, mask):
        """
        Args:
            left_gray: (B, 1, H, W) - for keypoint detection
            right_gray: (B, 1, H, W)
            left_rgb: (B, 3, H, W) - for DINOv3 feature extraction
            right_rgb: (B, 3, H, W)
            mask: (B, 1, H, W)
        Returns:
            dict with keypoints, descriptors, matches, disparity
        """
        B, _, H, W = left_gray.shape

        # 1. Detect keypoints
        kp_left, scores_left = self.keypoint_detector(left_gray, mask)
        kp_right, scores_right = self.keypoint_detector(right_gray, mask)

        # 2. Extract features
        desc_left = self.feature_extractor(left_rgb, kp_left)
        desc_right = self.feature_extractor(right_rgb, kp_right)

        # 3. Match keypoints
        match_scores, disparity = self.matcher(
            desc_left, desc_right, kp_left, kp_right, (H, W)
        )

        return {
            'keypoints_left': kp_left,
            'keypoints_right': kp_right,
            'scores_left': scores_left,
            'scores_right': scores_right,
            'descriptors_left': desc_left,
            'descriptors_right': desc_right,
            'match_scores': match_scores,
            'disparity': disparity
        }


# --- 7. Evaluation Metrics (独立类，不要嵌套!) ---
class EvaluationMetrics:
    @staticmethod
    def compute_sparse_metrics(disparity, keypoints_left, scores_left):
        """
        Compute metrics on sparse disparity predictions
        Args:
            disparity: (B, N) - predicted disparity
            keypoints_left: (B, N, 2)
            scores_left: (B, N) - keypoint validity scores
        Returns:
            dict of metrics
        """
        metrics = {}

        # Average disparity on valid keypoints
        valid_mask = scores_left > 0.1
        if valid_mask.sum() > 0:
            valid_disp = disparity[valid_mask]
            metrics['mean_disparity'] = valid_disp.mean().item()
            metrics['std_disparity'] = valid_disp.std().item()
            metrics['num_valid_keypoints'] = valid_mask.sum().item()
        else:
            metrics['mean_disparity'] = 0.0
            metrics['std_disparity'] = 0.0
            metrics['num_valid_keypoints'] = 0

        return metrics


# 全局函数，不要嵌套!
def collate_fn(batch):
    """Custom collate function to handle None samples"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# --- 8. Trainer (独立类，不要嵌套!) ---
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR, self.timestamp)
        for d in ["checkpoints", "visualizations", "logs", "tensorboard"]:
            os.makedirs(os.path.join(self.run_dir, d), exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        train_ds = RectifiedWaveStereoDataset(cfg, False)
        val_ds = RectifiedWaveStereoDataset(cfg, True)
        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_fn, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=0, pin_memory=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tensorboard")) if SummaryWriter else None
        self.model = SparseMatchingStereoModel(cfg).to(self.device)
        self.loss_fn = SparsePhotometricLoss(cfg)

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=cfg.LEARNING_RATE, weight_decay=1e-4)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-7
        )

        self.evaluator = EvaluationMetrics()
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION)
        self.step = 0
        self.log_file = os.path.join(self.run_dir, "logs", "training_log.json")
        self.loss_keys = ['total', 'photometric', 'smoothness']
        self.metric_keys = ['mean_disparity', 'std_disparity', 'num_valid_keypoints']
        self.history = {
            'train': {k: [] for k in self.loss_keys + self.metric_keys},
            'val': {k: [] for k in self.loss_keys + self.metric_keys}
        }

    def train(self):
        print("\n--- Starting Sparse Matching Training ---")
        print(f"Max keypoints: {self.cfg.MAX_KEYPOINTS}, Patch size: {self.cfg.PATCH_SIZE}x{self.cfg.PATCH_SIZE}")
        best_val_metric, epochs_no_improve = float('inf'), 0

        for epoch in range(self.cfg.NUM_EPOCHS):
            train_results = self._run_epoch(epoch, True)
            self._log_epoch_results('train', epoch, train_results)

            with torch.no_grad():
                val_results = self._run_epoch(epoch, False)
            self._log_epoch_results('val', epoch, val_results)

            val_loss = val_results.get('total', float('inf'))

            print(
                f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> "
                f"Train Loss: {train_results.get('total', 0.0):.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Keypoints: {val_results.get('num_valid_keypoints', 0):.0f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            self.update_log_file(epoch)
            if self.cfg.VISUALIZE_TRAINING:
                self.plot_training_history()

            if val_loss < best_val_metric:
                best_val_metric, epochs_no_improve = val_loss, 0
                torch.save(self.model.state_dict(),
                           os.path.join(self.run_dir, "checkpoints", "best_model_sparse.pth"))
                print(f"  Val loss improved to {best_val_metric:.4f}. Model saved.")
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
        """Pad to match DINOv3 patch size"""
        _, _, h, w = tensors[0].shape
        patch_size = 16
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
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

            # Move to device
            left_gray = data['left_gray'].to(self.device)
            right_gray = data['right_gray'].to(self.device)
            left_rgb = data['left_rgb'].to(self.device)
            right_rgb = data['right_rgb'].to(self.device)
            mask = data['mask'].to(self.device)

            # Pad inputs
            left_gray, right_gray, left_rgb, right_rgb, mask = self._pad_inputs(
                left_gray, right_gray, left_rgb, right_rgb, mask
            )

            with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION):
                # Forward pass
                outputs = self.model(left_gray, right_gray, left_rgb, right_rgb, mask)

                # Compute loss
                photo_loss, smooth_loss = self.loss_fn(
                    left_gray, right_gray,
                    outputs['keypoints_left'],
                    outputs['disparity'],
                    outputs['scores_left']
                )

                total_loss = (self.cfg.PHOTOMETRIC_WEIGHT * photo_loss +
                              self.cfg.SMOOTHNESS_WEIGHT * smooth_loss)

            if is_training:
                if torch.isfinite(total_loss):
                    accum_steps = self.cfg.GRADIENT_ACCUMULATION_STEPS
                    self.scaler.scale(total_loss / accum_steps).backward()
                    if (self.step + 1) % accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    print(f"\nWarning: Invalid loss at step {self.step}. Skipping.")

            # Compute metrics
            metrics = self.evaluator.compute_sparse_metrics(
                outputs['disparity'],
                outputs['keypoints_left'],
                outputs['scores_left']
            )

            # Accumulate results
            epoch_results['total'] += total_loss.item()
            epoch_results['photometric'] += photo_loss.item()
            epoch_results['smoothness'] += smooth_loss.item()
            for k in self.metric_keys:
                if k in metrics:
                    epoch_results[k] += metrics[k]

            pbar.set_postfix({
                'loss': total_loss.item(),
                'kpts': metrics.get('num_valid_keypoints', 0),
                'disp': metrics.get('mean_disparity', 0.0)
            })

            if is_training:
                if self.writer:
                    self.writer.add_scalar('Loss/step_train', total_loss.item(), self.step)

                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(data, outputs, self.step, "train")

                self.step += 1

        num_batches = len(loader)
        return {k: v / num_batches for k, v in epoch_results.items()} if num_batches > 0 else epoch_results

    def _log_epoch_results(self, phase, epoch, results):
        for k, v in results.items():
            self.history[phase][k].append(v)
            if self.writer:
                metric_type = 'Loss' if k in self.loss_keys else 'Metrics'
                self.writer.add_scalar(f"{metric_type}/{phase}_{k}", v, epoch)

    def visualize(self, data, outputs, step, phase):
        """Visualize sparse keypoints and matches"""
        left_gray = data['left_gray'][0, 0].cpu().numpy()  # (H, W)

        kp_left = outputs['keypoints_left'][0].cpu().detach().numpy()  # (N, 2)
        scores_left = outputs['scores_left'][0].cpu().detach().numpy()  # (N,)
        disparity = outputs['disparity'][0].cpu().detach().numpy()  # (N,)

        # Filter valid keypoints
        valid_mask = scores_left > 0.1
        kp_valid = kp_left[valid_mask]
        disp_valid = disparity[valid_mask]

        # Remove NaNs from disparity before plotting to prevent errors
        disp_valid = disp_valid[~np.isnan(disp_valid)]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Sparse Matching - Step: {step} ({phase})', fontsize=16)

        # Left: keypoints on image
        axes[0].imshow(left_gray, cmap='gray')
        if len(kp_valid) > 0:
            axes[0].scatter(kp_valid[:, 0], kp_valid[:, 1], c='red', s=20, alpha=0.6)
        axes[0].set_title(f"Detected Keypoints ({len(kp_valid)})")
        axes[0].axis('off')

        # Right: disparity distribution
        if len(disp_valid) > 0:
            axes[1].hist(disp_valid, bins=50, range=(0, np.maximum(1, disp_valid.max())), alpha=0.7, edgecolor='black')
            axes[1].axvline(disp_valid.mean(), color='red', linestyle='--',
                            label=f'Mean: {disp_valid.mean():.2f}')
            axes[1].set_xlabel('Disparity (pixels)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Disparity Distribution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.run_dir, "visualizations", f"{phase}_step_{step:06d}.png")
        plt.savefig(save_path, dpi=100)
        if self.writer:
            self.writer.add_figure(f'Visualization/{phase}', fig, step)
        plt.close(fig)

    def plot_training_history(self):
        if not self.history['train']['total']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sparse Matching Training History', fontsize=16)

        # Total Loss
        axes[0, 0].plot(self.history['train']['total'], label='Train Loss')
        axes[0, 0].plot(self.history['val']['total'], label='Val Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Loss")
        if self.history['train']['total'] and self.history['val']['total']:
            axes[0, 0].set_ylim(bottom=min(self.history['train']['total'] + self.history['val']['total']) - 0.01)

        # Number of keypoints
        axes[0, 1].plot(self.history['train']['num_valid_keypoints'], label='Train')
        axes[0, 1].plot(self.history['val']['num_valid_keypoints'], label='Val')
        axes[0, 1].set_title('Number of Valid Keypoints')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_xlabel("Epochs")
        axes[0, 1].set_ylabel("Count")

        # Mean disparity
        axes[1, 0].plot(self.history['train']['mean_disparity'], label='Train')
        axes[1, 0].plot(self.history['val']['mean_disparity'], label='Val')
        axes[1, 0].set_title('Mean Disparity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Disparity (pixels)")

        # Loss components
        axes[1, 1].plot(self.history['train']['photometric'], label='Photometric')
        axes[1, 1].plot(self.history['train']['smoothness'], label='Smoothness')
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


# 主程序入口 (不要缩进!)
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

