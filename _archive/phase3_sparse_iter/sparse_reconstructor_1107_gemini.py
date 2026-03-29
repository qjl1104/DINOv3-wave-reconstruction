# sparse_matching_stereo.py (v2 - Feature-Metric Loss)
# FINAL TRAINING SCRIPT (v19 - JSON Log Fix)
#
# This script is now READY FOR TRAINING.
# 1. (Phase 1) DualDetector: Separate detector params for Left vs Right (TUNED).
# 2. (Phase 1) Asymmetric Augmentation: In Dataset, apply independent augs.
# 3. (Phase 1) Dual Mask: Correctly generate and use a mask for the right image.
# 4. (Phase 2) HybridLoss: Implement Photometric L1 loss alongside Feature/Smoothness.
# 5. (FIX v19): Fixed "Circular reference detected" in update_log_file.

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

    # --- Auto-detection ---
    IMAGE_HEIGHT: int = 0
    IMAGE_WIDTH: int = 0

    MASK_THRESHOLD: int = 30  # Base threshold for mask generation

    # Sparse matching settings
    MAX_KEYPOINTS: int = 512
    PATCH_SIZE: int = 5  # Patch size for NEW photometric loss
    FEATURE_DIM: int = 768
    NUM_ATTENTION_LAYERS: int = 4
    NUM_HEADS: int = 8
    DISPARITY_CONSTRAINT_Y_THRESHOLD: int = 2
    MATCHING_TEMPERATURE: float = 10.0

    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    GRADIENT_ACCUMULATION_STEPS: int = 1
    USE_MIXED_PRECISION: bool = True

    # --- (Phase 2) Hybrid Loss Weights ---
    FEATURE_WEIGHT: float = 1.0  # (Renamed from PHOTOMETRIC_WEIGHT)
    SMOOTHNESS_WEIGHT: float = 0.0001
    PHOTOMETRIC_WEIGHT: float = 0.5  # (NEW loss weight)

    # --- (Phase 1) Asymmetric Augmentation ---
    USE_ASYMMETRIC_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.8
    AUG_BRIGHTNESS_RANGE: tuple = (0.7, 1.3)
    AUG_CONTRAST_RANGE: tuple = (0.7, 1.3)
    AUG_NOISE_STD: float = 5.0
    AUG_OCCLUSION_PROB: float = 0.2
    AUG_OCCLUSION_SIZE: tuple = (20, 100)

    EARLY_STOPPING_PATIENCE: int = 25


def auto_tune_config(cfg: Config):
    # This function is now SKIPPED
    pass


# =========================================================================
# --- [START] REPLACEMENT OF SparseKeypointDetector (v17 - Dual Detector) ---
# =========================================================================
class SparseKeypointDetector(nn.Module):
    """
    Detect sparse keypoints using OpenCV's SimpleBlobDetector.
    Accepts a 'camera_side' argument to initialize
    different parameters for left and right cameras.
    """

    def __init__(self, cfg: Config, camera_side: str = 'left'):
        super().__init__()
        self.cfg = cfg
        self.max_keypoints = cfg.MAX_KEYPOINTS

        # --- Setup Blob Detector ---
        params = cv2.SimpleBlobDetector_Params()

        # --- Use Thresholding (Robust to illumination) ---
        params.filterByColor = False
        params.thresholdStep = 10

        # --- Shape Filters (Relaxed for ellipses) ---
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.1

        # --- (Phase 1) DUAL PARAMETERS ---
        if camera_side == 'left':
            # Parameters based on v15 test (363 points)
            # Catches both large and small points
            print(f"--- Initializing LEFT Detector (v15 params) ---")
            params.minThreshold = 30
            params.maxThreshold = 255
            params.filterByArea = True
            params.minArea = 10
            params.maxArea = 5000
            params.filterByCircularity = True
            params.minCircularity = 0.3
        else:
            # Parameters for RIGHT camera (User suggestion from Phase 1 plan)
            print(f"--- Initializing RIGHT Detector (New params) ---")
            params.minThreshold = 50
            params.maxThreshold = 255
            params.filterByArea = True
            params.minArea = 15
            params.maxArea = 5000  # Keep high
            params.filterByCircularity = True
            params.minCircularity = 0.4
        # --- END DUAL PARAMETERS ---

        # Create detector
        self.detector = cv2.SimpleBlobDetector_create(params)
        print(
            f"[{camera_side.upper()}] Thresholds: [{params.minThreshold}, {params.maxThreshold}], Step: {params.thresholdStep}")
        print(
            f"[{camera_side.upper()}] Filter by Area: {params.filterByArea}, Range: [{params.minArea}, {params.maxArea}]")
        print(
            f"[{camera_side.upper()}] Filter by Circularity: {params.filterByCircularity}, Min: {params.minCircularity}")
        print(f"-------------------------------------------------------")
        # --- End Setup ---

    def forward(self, gray_image, mask):
        """
        Args:
            gray_image: (B, 1, H, W) - grayscale image (0-1 range)
            mask: (B, 1, H, W) - valid region mask
        Returns:
            keypoints: (B, N, 2) - (x, y) coordinates, N <= max_keypoints
            scores: (B, N) - blob size (used as score)
        """
        B, _, H, W = gray_image.shape
        device = gray_image.device

        keypoints_list = []
        scores_list = []

        for b in range(B):
            # Convert tensor to numpy uint8 image
            img_tensor = gray_image[b, 0]  # (H, W)
            mask_tensor = mask[b, 0]  # (H, W)

            # Apply mask
            img_masked_tensor = img_tensor * mask_tensor

            # Convert from (0.0 - 1.0) float to (0 - 255) uint8
            img_np = (img_masked_tensor.cpu().numpy() * 255).astype(np.uint8)

            # Detect blobs
            cv_keypoints = self.detector.detect(img_np)

            if not cv_keypoints:
                keypoints_list.append(torch.zeros(1, 2, device=device))
                scores_list.append(torch.zeros(1, device=device))
                continue

            # Extract coords and scores (use blob size as score)
            kp_coords = np.array([kp.pt for kp in cv_keypoints]).astype(np.float32)
            kp_scores = np.array([kp.size for kp in cv_keypoints]).astype(np.float32)
            kp_tensor = torch.from_numpy(kp_coords).to(device)
            scores_tensor = torch.from_numpy(kp_scores).to(device)

            # Limit to max_keypoints
            if len(kp_tensor) > self.max_keypoints:
                sorted_indices = torch.argsort(scores_tensor, descending=True)
                kp_tensor = kp_tensor[sorted_indices[:self.max_keypoints]]
                scores_tensor = scores_tensor[sorted_indices[:self.max_keypoints]]

            keypoints_list.append(kp_tensor)
            scores_list.append(scores_tensor)

        # Pad to same length
        max_len = max(len(kp) for kp in keypoints_list)
        if max_len == 0: max_len = 1

        keypoints_padded = []
        scores_padded = []
        for kp, sc in zip(keypoints_list, scores_list):
            if len(kp) == 0:
                kp = torch.zeros(1, 2, device=device)
                sc = torch.zeros(1, device=device)
            pad_len = max_len - len(kp)
            if pad_len > 0:
                kp_pad = torch.cat([kp, torch.zeros(pad_len, 2, device=device)], dim=0)
                sc_pad = torch.cat([sc, torch.zeros(pad_len, device=device)], dim=0)
            else:
                kp_pad, sc_pad = kp, sc
            keypoints_padded.append(kp_pad)
            scores_padded.append(sc_pad)

        keypoints = torch.stack(keypoints_padded, dim=0)
        scores = torch.stack(scores_padded, dim=0)

        return keypoints, scores


# =========================================================================
# --- [END] REPLACEMENT OF SparseKeypointDetector ---
# =========================================================================


# --- 2. DINOv3 Feature Extractor (unchanged) ---
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
        with torch.no_grad():
            features = self.dino(image).last_hidden_state
        b, _, h, w = image.shape
        start_idx = 1 + self.num_register_tokens
        patch_tokens = features[:, start_idx:, :]
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        feat_map = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, h_feat, w_feat)
        return feat_map

    def forward(self, image, keypoints):
        B, N, _ = keypoints.shape
        feat_map = self.get_feature_map(image)
        _, C, H_feat, W_feat = feat_map.shape
        _, _, H_img, W_img = image.shape
        grid = keypoints.clone()
        grid[:, :, 0] = 2 * (grid[:, :, 0] / W_img) - 1
        grid[:, :, 1] = 2 * (grid[:, :, 1] / H_img) - 1
        grid = grid.unsqueeze(2)
        descriptors = F.grid_sample(
            feat_map, grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        ).squeeze(3).permute(0, 2, 1)
        return descriptors


# --- 3. Attention-based Sparse Matching Network (unchanged) ---
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(2, dim)

    def forward(self, positions, image_size):
        H, W = image_size
        pos_normalized = positions.clone()
        pos_normalized[:, :, 0] = pos_normalized[:, :, 0] / W
        pos_normalized[:, :, 1] = pos_normalized[:, :, 1] / H
        return self.proj(pos_normalized)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, features, pos_enc):
        features_pos = features + pos_enc
        attn_out, _ = self.attn(features_pos, features_pos, features_pos)
        features = self.norm1(features + attn_out)
        ffn_out = self.ffn(features)
        features = self.norm2(features + ffn_out)
        return features


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_query, feat_kv):
        attn_out, attn_weights = self.attn(feat_query, feat_kv, feat_kv)
        return self.norm(feat_query + attn_out), attn_weights


class SparseMatchingNetwork(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim = cfg.FEATURE_DIM
        num_layers = cfg.NUM_ATTENTION_LAYERS
        num_heads = cfg.NUM_HEADS
        self.pos_enc = PositionalEncoding(dim)
        self.self_attn_left = nn.ModuleList([
            SelfAttentionLayer(dim, num_heads) for _ in range(num_layers)
        ])
        self.self_attn_right = nn.ModuleList([
            SelfAttentionLayer(dim, num_heads) for _ in range(num_layers)
        ])
        self.cross_attn = CrossAttentionLayer(dim, num_heads)

    def forward(self, desc_left, desc_right, kp_left, kp_right, image_size):
        pos_left = self.pos_enc(kp_left, image_size)
        pos_right = self.pos_enc(kp_right, image_size)
        feat_left, feat_right = desc_left, desc_right
        for self_l, self_r in zip(self.self_attn_left, self.self_attn_right):
            feat_left = self_l(feat_left, pos_left)
            feat_right = self_r(feat_right, pos_right)
        feat_left_enhanced, attn_weights = self.cross_attn(feat_left, feat_right)
        eps = 1e-8
        feat_left_norm = F.normalize(feat_left_enhanced, dim=2, eps=eps)
        feat_right_norm = F.normalize(feat_right, dim=2, eps=eps)
        match_scores = torch.bmm(feat_left_norm, feat_right_norm.transpose(1, 2))
        x_left = kp_left[:, :, 0].unsqueeze(2)
        y_left = kp_left[:, :, 1].unsqueeze(2)
        x_right = kp_right[:, :, 0].unsqueeze(1)
        y_right = kp_right[:, :, 1].unsqueeze(1)
        valid_x_mask = (x_left >= x_right)
        valid_y_mask = (y_left - y_right).abs() < self.cfg.DISPARITY_CONSTRAINT_Y_THRESHOLD
        constraint_mask = (valid_x_mask & valid_y_mask)
        match_scores_constrained = match_scores.masked_fill(~constraint_mask, torch.finfo(match_scores.dtype).min)
        match_probs = F.softmax(match_scores_constrained * self.cfg.MATCHING_TEMPERATURE, dim=2)
        disparity_matrix = kp_left[:, :, 0].unsqueeze(2) - kp_right[:, :, 0].unsqueeze(1)
        disparity_pred = (disparity_matrix * match_probs).sum(dim=2)
        disparity_pred = torch.nan_to_num(disparity_pred)
        return match_scores_constrained, disparity_pred, constraint_mask


# =========================================================================
# --- [START] REPLACEMENT OF FeatureMetricLoss (v17 - Hybrid Loss) ---
# =========================================================================
class HybridLoss(nn.Module):
    """
    (Phase 2) Computes a hybrid loss:
    1. Feature Loss: Cosine similarity between DINOv3 descriptors (original)
    2. Smoothness Loss: Sparse smoothness (original)
    3. Photometric Loss: L1 pixel difference on warped patches (NEW)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.PATCH_SIZE
        self.patch_radius = self.patch_size // 2

    def _get_normalized_grid(self, kp, H, W):
        """Normalize keypoint coordinates to [-1, 1] for grid_sample"""
        grid = kp.clone()
        grid[:, :, 0] = 2 * (grid[:, :, 0] / W) - 1  # x to [-1, 1]
        grid[:, :, 1] = 2 * (grid[:, :, 1] / H) - 1  # y to [-1, 1]
        return grid

    def _sample_patches(self, image, kp_coords_norm, B, N, H, W):
        """
        Sample patches from an image using grid_sample.
        This is tricky. We'll sample pixel by pixel for simplicity.
        """
        patches = []
        for dy in range(-self.patch_radius, self.patch_radius + 1):
            for dx in range(-self.patch_radius, self.patch_radius + 1):
                # Calculate offsets in normalized space
                offset_x = 2 * dx / W
                offset_y = 2 * dy / H

                # Create grid of sample points
                grid = kp_coords_norm.clone()
                grid[:, :, 0] += offset_x
                grid[:, :, 1] += offset_y

                # Sample one pixel from the grid
                pixel_sample = F.grid_sample(
                    image, grid.unsqueeze(2),  # (B, N, 1, 2)
                    mode='bilinear',
                    align_corners=True,
                    padding_mode='border'
                ).squeeze(3)  # (B, 1, N)
                patches.append(pixel_sample.transpose(1, 2))  # (B, N, 1)

        # Stack patches
        patches_stacked = torch.cat(patches, dim=2)  # (B, N, P*P)
        return patches_stacked.view(B, N, self.patch_size, self.patch_size)

    def _compute_photometric_loss(self, left_gray, right_gray, kp_left, disparity, valid_mask):
        """
        Compute L1 loss on patches warped by disparity.
        """
        B, _, H, W = left_gray.shape
        N = kp_left.shape[1]
        device = left_gray.device

        # 1. Get coordinates for left patches
        kp_left_norm = self._get_normalized_grid(kp_left, H, W)  # (B, N, 2)

        # 2. Calculate coordinates for right patches
        kp_right_warped = kp_left.clone()

        # --- [FIX v18] ---
        # Original line: kp_right_warped[:, :, 0] -= disparity.unsqueeze(2)
        #   LHS shape: [B, N]
        #   RHS shape: [B, N, 1] -> Broadcasting error
        # Corrected line:
        kp_right_warped[:, :, 0] -= disparity  # (B, N)
        # --- [END FIX v18] ---

        kp_right_warped_norm = self._get_normalized_grid(kp_right_warped, H, W)  # (B, N, 2)

        # 3. Sample patches
        # We'll just sample the center pixel for simplicity and speed.
        patch_left = F.grid_sample(
            left_gray, kp_left_norm.unsqueeze(2),
            mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(2)  # (B, 1, N)

        patch_right_warped = F.grid_sample(
            right_gray, kp_right_warped_norm.unsqueeze(2),
            mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(2)  # (B, 1, N)

        # 4. Compute L1 loss
        photometric_l1 = (patch_left - patch_right_warped).abs()  # (B, 1, N)

        # 5. Mask and average
        masked_loss = photometric_l1.squeeze(1) * valid_mask  # (B, N)

        photometric_loss = torch.tensor(0.0, device=device)
        if valid_mask.sum() > 0:
            photometric_loss = masked_loss.sum() / valid_mask.sum()

        return photometric_loss

    def _compute_feature_loss(self, desc_left, desc_right, match_scores, constraint_mask, valid_mask):
        """
        Compute cosine similarity loss on DINO descriptors.
        """
        device = desc_left.device

        # We use the 'match_scores' which already have epipolar constraints applied
        match_probs = F.softmax(match_scores * self.cfg.MATCHING_TEMPERATURE, dim=2)  # (B, N_l, N_r)
        match_probs = torch.nan_to_num(match_probs, nan=0.0)

        # Get weighted average of right descriptors
        desc_right_weighted = torch.bmm(match_probs, desc_right)

        eps = 1e-8
        desc_left_norm = F.normalize(desc_left, dim=2, eps=eps)
        desc_right_weighted_norm = F.normalize(desc_right_weighted, dim=2, eps=eps)

        # Calculate cosine similarity loss (1 - sim)
        cosine_sim = (desc_left_norm * desc_right_weighted_norm).sum(dim=2)  # (B, N_l)
        feature_loss_per_kp = 1.0 - cosine_sim  # (B, N_l)

        # Mask out invalid keypoints
        masked_loss = feature_loss_per_kp * valid_mask

        feature_loss = torch.tensor(0.0, device=device)
        if valid_mask.sum() > 0:
            feature_loss = masked_loss.sum() / valid_mask.sum()

        return feature_loss

    def _compute_sparse_smoothness(self, keypoints, disparity, scores, valid_mask):
        """Smoothness: nearby keypoints should have similar disparity"""
        B, N, _ = keypoints.shape
        smooth_loss = 0.0
        valid_pairs_total = 0

        for b in range(B):
            kp_valid = keypoints[b][valid_mask[b]]
            disp_valid = disparity[b][valid_mask[b]]

            if len(kp_valid) < 2:
                continue

            # Compute pairwise distances
            dist = torch.cdist(kp_valid, kp_valid)
            neighbor_mask = (dist < 20) & (dist > 0)

            if neighbor_mask.sum() == 0:
                continue

            disp_diff = (disp_valid.unsqueeze(0) - disp_valid.unsqueeze(1)).abs()
            smooth_loss += (disp_diff * neighbor_mask.float()).sum() / neighbor_mask.sum()
            valid_pairs_total += 1

        if valid_pairs_total > 0:
            smooth_loss /= valid_pairs_total

        return smooth_loss

    def forward(self, left_gray, right_gray, desc_left, desc_right, match_scores,
                keypoints_left, disparity, scores_left, constraint_mask):
        """
        Args:
            left_gray, right_gray: (B, 1, H, W) PADDED gray images
            desc_left, desc_right: (B, N, C) Descriptors
            match_scores: (B, N_l, N_r) Match scores (pre-softmax, constrained)
            keypoints_left: (B, N_l, 2)
            disparity: (B, N_l)
            scores_left: (B, N_l)
            constraint_mask: (B, N_l, N_r) Mask of valid matches
        """

        # 1. Create the final valid mask
        detection_mask = (scores_left > 0.1)
        matchable_mask = torch.any(constraint_mask, dim=2)
        final_valid_mask = detection_mask & matchable_mask  # (B, N_l)

        # 2. Feature Loss
        feature_loss = self._compute_feature_loss(
            desc_left, desc_right, match_scores, constraint_mask, final_valid_mask
        )

        # 3. Smoothness Loss
        smooth_loss = self._compute_sparse_smoothness(
            keypoints_left, disparity, scores_left, final_valid_mask
        )

        # 4. Photometric Loss (NEW)
        photometric_loss = self._compute_photometric_loss(
            left_gray, right_gray, keypoints_left, disparity, final_valid_mask
        )

        return feature_loss, smooth_loss, photometric_loss


# =========================================================================
# --- [END] REPLACEMENT OF FeatureMetricLoss ---
# =========================================================================


# =========================================================================
# --- [START] MODIFICATION OF Dataset (v17 - Asymmetric Aug, Dual Mask) ---
# =========================================================================
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

            _x_l, _y_l, w_l, h_l = self.roi_left
            _x_r, _y_r, w_r, h_r = self.roi_right

            if self.cfg.IMAGE_WIDTH == 0 or self.cfg.IMAGE_HEIGHT == 0:
                print(f"[Dataset] Left ROI: {w_l}x{h_l}, Right ROI: {w_r}x{h_r}")
                target_w = min(w_l, w_r)
                target_h = min(h_l, h_r)
                print(f"[Dataset] Setting target config resolution to MINIMUM common ROI: {target_w}x{target_h}")
                self.cfg.IMAGE_WIDTH = target_w
                self.cfg.IMAGE_HEIGHT = target_h

        except Exception as e:
            sys.exit(f"Failed to load calibration file: {e}")

        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self):
        return len(self.indices)

    def _apply_asymmetric_augmentation(self, left_img, right_img):
        """(Phase 1) Apply asymmetric augmentations to simulate camera differences"""
        cfg = self.cfg

        # 1. Asymmetric Brightness/Contrast
        if np.random.rand() < 0.5:
            brightness_l = np.random.uniform(*cfg.AUG_BRIGHTNESS_RANGE)
            brightness_r = np.random.uniform(*cfg.AUG_BRIGHTNESS_RANGE)
            left_img = np.clip(left_img * brightness_l, 0, 255)
            right_img = np.clip(right_img * brightness_r, 0, 255)

        if np.random.rand() < 0.5:
            contrast_l = np.random.uniform(*cfg.AUG_CONTRAST_RANGE)
            contrast_r = np.random.uniform(*cfg.AUG_CONTRAST_RANGE)
            left_img = np.clip((left_img - left_img.mean()) * contrast_l + left_img.mean(), 0, 255)
            right_img = np.clip((right_img - right_img.mean()) * contrast_r + right_img.mean(), 0, 255)

        # 2. Independent Noise
        if np.random.rand() < 0.3:
            noise_left = np.random.normal(0, cfg.AUG_NOISE_STD, left_img.shape)
            noise_right = np.random.normal(0, cfg.AUG_NOISE_STD, right_img.shape)
            left_img = np.clip(left_img + noise_left, 0, 255)
            right_img = np.clip(right_img + noise_right, 0, 255)

        # 3. One-sided Occlusion
        if np.random.rand() < cfg.AUG_OCCLUSION_PROB:
            h, w = left_img.shape
            size = np.random.randint(*cfg.AUG_OCCLUSION_SIZE)
            x1 = np.random.randint(0, w - size)
            y1 = np.random.randint(0, h - size)

            if np.random.rand() < 0.5:
                left_img[y1:y1 + size, x1:x1 + size] = 0  # Occlude left
            else:
                right_img[y1:y1 + size, x1:x1 + size] = 0  # Occlude right

        return left_img.astype(np.uint8), right_img.astype(np.uint8)

    def __getitem__(self, idx):
        try:
            left_path = self.left_images[self.indices[idx]]
            right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, 'right' + os.path.basename(left_path)[4:])
            left_raw, right_raw = cv2.imread(left_path, 0), cv2.imread(right_path, 0)
            if left_raw is None or right_raw is None: return None

            left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

            x, y, w, h = self.roi_left
            left_rect = left_rect[y:y + h, x:x + w]
            x, y, w, h = self.roi_right
            right_rect = right_rect[y:y + h, x:x + w]

            # CROP both images to the minimum common size
            left_img = left_rect[0:self.cfg.IMAGE_HEIGHT, 0:self.cfg.IMAGE_WIDTH]
            right_img = right_rect[0:self.cfg.IMAGE_HEIGHT, 0:self.cfg.IMAGE_WIDTH]

            if left_img.shape[0] != self.cfg.IMAGE_HEIGHT or right_img.shape[0] != self.cfg.IMAGE_HEIGHT:
                print(f"[ERROR] Image shape mismatch after crop! Skipping index {idx}")
                return None

            # Augmentation
            if not self.is_validation and self.cfg.USE_ASYMMETRIC_AUGMENTATION and \
                    np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
                left_img, right_img = self._apply_asymmetric_augmentation(left_img, right_img)

            # --- (Phase 1) Create DUAL MASKS ---
            _, mask_left = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
            _, mask_right = cv2.threshold(right_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
            # --- END DUAL MASKS ---

            # Convert to tensors
            left_gray = torch.from_numpy(left_img).float().unsqueeze(0) / 255.0
            right_gray = torch.from_numpy(right_img).float().unsqueeze(0) / 255.0
            left_rgb = torch.from_numpy(cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0
            right_rgb = torch.from_numpy(cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0
            mask_left_tensor = torch.from_numpy(mask_left).float().unsqueeze(0) / 255.0
            mask_right_tensor = torch.from_numpy(mask_right).float().unsqueeze(0) / 255.0

            return {
                'left_gray': left_gray,
                'right_gray': right_gray,
                'left_rgb': left_rgb,
                'right_rgb': right_rgb,
                'mask_left': mask_left_tensor,
                'mask_right': mask_right_tensor
            }
        except Exception as e:
            print(f"Warning at idx {idx}: {e}")
            import traceback
            traceback.print_exc()
            return None


# =========================================================================
# --- [END] MODIFICATION OF Dataset ---
# =========================================================================


# =========================================================================
# --- [START] MODIFICATION OF Model (v17 - Dual Detector, Dual Mask) ---
# =========================================================================
class SparseMatchingStereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # --- (Phase 1) Create DUAL detectors ---
        self.keypoint_detector_left = SparseKeypointDetector(cfg, camera_side='left')
        self.keypoint_detector_right = SparseKeypointDetector(cfg, camera_side='right')
        # --- END DUAL detectors ---

        self.feature_extractor = DINOv3FeatureExtractor(cfg)
        self.matcher = SparseMatchingNetwork(cfg)
        print(f"Sparse Matching Model: max {cfg.MAX_KEYPOINTS} keypoints, "
              f"{cfg.NUM_ATTENTION_LAYERS} attention layers")
        print(f"Using NEW Hybrid Loss")

    def forward(self, left_gray, right_gray, left_rgb, right_rgb, mask_left, mask_right):
        """
        Args:
            left_gray, right_gray: (B, 1, H, W) - for keypoint detection
            left_rgb, right_rgb: (B, 3, H, W) - for DINOv3 feature extraction
            mask_left, mask_right: (B, 1, H, W) - DUAL MASKS
        """
        B, _, H, W = left_gray.shape

        # 1. Detect keypoints (using respective detectors and masks)
        kp_left, scores_left = self.keypoint_detector_left(left_gray, mask_left)
        kp_right, scores_right = self.keypoint_detector_right(right_gray, mask_right)

        # 2. Extract features
        desc_left = self.feature_extractor(left_rgb, kp_left)
        desc_right = self.feature_extractor(right_rgb, kp_right)

        # 3. Match keypoints
        match_scores, disparity, constraint_mask = self.matcher(
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
            'disparity': disparity,
            'constraint_mask': constraint_mask
        }


# =========================================================================
# --- [END] MODIFICATION OF Model ---
# =========================================================================


# --- 7. Evaluation Metrics (unchanged) ---
class EvaluationMetrics:
    @staticmethod
    def compute_sparse_metrics(disparity, keypoints_left, scores_left):
        metrics = {}
        valid_mask = scores_left > 0.1
        if valid_mask.sum() > 0:
            valid_disp = disparity[valid_mask]
            valid_disp_finite = valid_disp[torch.isfinite(valid_disp)]
            if valid_disp_finite.numel() > 0:
                metrics['mean_disparity'] = valid_disp_finite.mean().item()
                metrics['std_disparity'] = valid_disp_finite.std().item()
            else:
                metrics['mean_disparity'] = 0.0
                metrics['std_disparity'] = 0.0
            metrics['num_valid_keypoints'] = valid_mask.sum().item()
        else:
            metrics['mean_disparity'] = 0.0
            metrics['std_disparity'] = 0.0
            metrics['num_valid_keypoints'] = 0
        return metrics


# --- Collate Function (unchanged) ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# --- 8. Trainer (Modified for Hybrid Loss) ---
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

        print(f"--- Using FINAL resolution (min common ROI): {self.cfg.IMAGE_WIDTH}x{self.cfg.IMAGE_HEIGHT} ---")

        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_fn, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=0, pin_memory=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tensorboard")) if SummaryWriter else None
        self.model = SparseMatchingStereoModel(cfg).to(self.device)

        # --- (Phase 2) Use HybridLoss ---
        self.loss_fn = HybridLoss(cfg)

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=cfg.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-7
        )
        self.evaluator = EvaluationMetrics()
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION)
        self.step = 0
        self.log_file = os.path.join(self.run_dir, "logs", "training_log.json")

        # --- (Phase 2) Update loss keys ---
        self.loss_keys = ['total', 'feature', 'smoothness', 'photometric']
        self.metric_keys = ['mean_disparity', 'std_disparity', 'num_valid_keypoints']
        self.history = {
            'train': {k: [] for k in self.loss_keys + self.metric_keys},
            'val': {k: [] for k in self.loss_keys + self.metric_keys}
        }

    def train(self):
        # --- [FINAL] Re-enabling train() ---
        print("\n--- Starting Sparse Matching Training (v18: Hybrid Loss) ---")
        print(f"Max keypoints: {self.cfg.MAX_KEYPOINTS}, Loss Temp: {self.cfg.MATCHING_TEMPERATURE}")
        print(
            f"Loss Weights: Feature={self.cfg.FEATURE_WEIGHT}, Smooth={self.cfg.SMOOTHNESS_WEIGHT}, Photo={self.cfg.PHOTOMETRIC_WEIGHT}")
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
        # --- [FINAL] End re-enabling ---

    # --- (Phase 1) Padding function (FIXED) ---
    def _pad_inputs(self, *tensors):
        patch_size = 16
        padded_tensors = []
        target_h, target_w = -1, -1
        for i, x in enumerate(tensors):
            if x is None:
                padded_tensors.append(None)
                continue
            _, _, h, w = x.shape
            if i == 0:
                pad_h = (patch_size - h % patch_size) % patch_size
                pad_w = (patch_size - w % patch_size) % patch_size
                target_h = h + pad_h
                target_w = w + pad_w
            current_pad_h = max(0, target_h - h)
            current_pad_w = max(0, target_w - w)
            if current_pad_h > 0 or current_pad_w > 0:
                padding = (0, current_pad_w, 0, current_pad_h)
                try:
                    padded_x = F.pad(x, padding, mode='constant', value=0)
                    if padded_x.shape[2] != target_h or padded_x.shape[3] != target_w:
                        print(f"[ERROR] Padding failed for tensor {i}.")
                        padded_tensors.append(x)
                    else:
                        padded_tensors.append(padded_x)
                except Exception as e:
                    print(f"[ERROR] Exception during F.pad for tensor {i}: {e}")
                    padded_tensors.append(x)
            else:
                padded_tensors.append(x)
        return padded_tensors

    def _run_epoch(self, epoch, is_training):
        # --- THIS IS THE REAL TRAINING LOOP ---
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [{'Train' if is_training else 'Val'}]")
        epoch_results = {k: 0.0 for k in self.loss_keys}
        for k in self.metric_keys: epoch_results[k] = 0.0

        for data in pbar:
            if data is None:
                continue

            # --- (Phase 1) Load DUAL MASKS ---
            left_gray = data['left_gray'].to(self.device)
            right_gray = data['right_gray'].to(self.device)
            left_rgb = data['left_rgb'].to(self.device)
            right_rgb = data['right_rgb'].to(self.device)
            mask_left = data['mask_left'].to(self.device)
            mask_right = data['mask_right'].to(self.device)

            # Pad inputs
            padded_inputs = self._pad_inputs(
                left_gray, right_gray, left_rgb, right_rgb, mask_left, mask_right
            )
            left_gray_pad, right_gray_pad, left_rgb_pad, right_rgb_pad, mask_left_pad, mask_right_pad = padded_inputs

            if None in padded_inputs:
                print(f"[ERROR] Padding returned None. Skipping batch.")
                continue

            with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION):
                # Forward pass
                outputs = self.model(
                    left_gray_pad, right_gray_pad, left_rgb_pad, right_rgb_pad,
                    mask_left_pad, mask_right_pad
                )

                # --- (Phase 2) Call HybridLoss ---
                feature_loss, smooth_loss, photometric_loss = self.loss_fn(
                    left_gray_pad, right_gray_pad,  # Pass images for photometric loss
                    outputs['descriptors_left'],
                    outputs['descriptors_right'],
                    outputs['match_scores'],
                    outputs['keypoints_left'],
                    outputs['disparity'],
                    outputs['scores_left'],
                    outputs['constraint_mask']
                )

                total_loss = (self.cfg.FEATURE_WEIGHT * feature_loss +
                              self.cfg.SMOOTHNESS_WEIGHT * smooth_loss +
                              self.cfg.PHOTOMETRIC_WEIGHT * photometric_loss)
                # --- End HybridLoss ---

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
                outputs['disparity'], outputs['keypoints_left'], outputs['scores_left']
            )

            # Accumulate results
            epoch_results['total'] += total_loss.item() if torch.isfinite(total_loss) else 0.0
            epoch_results['feature'] += feature_loss.item() if torch.isfinite(feature_loss) else 0.0
            epoch_results['smoothness'] += smooth_loss.item() if torch.isfinite(smooth_loss) else 0.0
            epoch_results['photometric'] += photometric_loss.item() if torch.isfinite(photometric_loss) else 0.0

            for k in self.metric_keys:
                if k in metrics: epoch_results[k] += metrics[k]

            pbar.set_postfix({
                'loss': total_loss.item(),
                'feat': feature_loss.item(),
                'photo': photometric_loss.item(),
                'kpts': metrics.get('num_valid_keypoints', 0)
            })

            if is_training:
                if self.writer:
                    self.writer.add_scalar('Loss/step_train', total_loss.item(), self.step)
                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(data, outputs, self.step, "train")
                self.step += 1

        num_batches = len(loader)
        if num_batches > 0:
            return {k: v / num_batches for k, v in epoch_results.items()}
        return epoch_results

    def _log_epoch_results(self, phase, epoch, results):
        for k, v in results.items():
            self.history[phase][k].append(v)
            if self.writer:
                if k == 'smoothness':
                    self.writer.add_scalar(f"Loss_Unweighted/{phase}_smoothness", v, epoch)
                elif k in self.loss_keys:
                    self.writer.add_scalar(f"Loss/{phase}_{k}", v, epoch)
                else:
                    self.writer.add_scalar(f"Metrics/{phase}_{k}", v, epoch)

    # =========================================================================
    # --- [START] VISUALIZATION FOR TRAINING (NOT CHECK MODE) ---
    # =========================================================================
    def visualize(self, data, outputs, step, phase):
        """
        Visualize sparse keypoints and matches
        """
        # --- Get Left Image Data ---
        left_gray_orig = data['left_gray'][0, 0].cpu().numpy()  # (H_orig, W_orig)
        kp_left = outputs['keypoints_left'][0].cpu().detach().numpy()
        scores_left = outputs['scores_left'][0].cpu().detach().numpy()
        disparity = outputs['disparity'][0].cpu().detach().numpy()

        # --- Get Right Image Data ---
        right_gray_orig = data['right_gray'][0, 0].cpu().numpy()
        kp_right = outputs['keypoints_right'][0].cpu().detach().numpy()
        scores_right = outputs['scores_right'][0].cpu().detach().numpy()

        # --- Filter valid Left keypoints ---
        valid_mask_left = scores_left > 0.1
        kp_left_valid = kp_left[valid_mask_left]
        disp_valid = disparity[valid_mask_left]

        # --- Filter valid Right keypoints ---
        valid_mask_right = scores_right > 0.1
        kp_right_valid = kp_right[valid_mask_right]

        # Remove NaNs from disparity before plotting
        if len(disp_valid) > 0:
            finite_mask = np.isfinite(disp_valid)
            disp_valid = disp_valid[finite_mask]
            kp_left_valid = kp_left_valid[finite_mask]

        # --- Create Plot ---
        h, w = left_gray_orig.shape
        fig_width = 15
        fig_height = (h / w) * fig_width * 0.5 if w > 0 else fig_width * 0.5 * (3 / 4)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height + 1))

        fig.suptitle(f'Training Visualization - Step: {step} ({phase})', fontsize=16)

        # --- Left: keypoints on image (colored by disparity) ---
        axes[0].imshow(left_gray_orig, cmap='gray')
        if len(kp_left_valid) > 0:
            # Use disparity for color
            sc = axes[0].scatter(kp_left_valid[:, 0], kp_left_valid[:, 1], c=disp_valid,
                                 cmap='viridis', s=20, alpha=0.7,
                                 vmin=0, vmax=np.percentile(disp_valid, 95) if len(disp_valid) > 0 else 50)
            plt.colorbar(sc, ax=axes[0], label='Disparity (pixels)')
        axes[0].set_title(f"Detected Left Keypoints ({len(kp_left_valid)})")
        axes[0].axis('off')

        # --- Right: keypoints on image ---
        axes[1].imshow(right_gray_orig, cmap='gray')
        if len(kp_right_valid) > 0:
            axes[1].scatter(kp_right_valid[:, 0], kp_right_valid[:, 1], c='cyan', s=20, alpha=0.6)
        axes[1].set_title(f"Detected Right Keypoints ({len(kp_right_valid)})")
        axes[1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.run_dir, "visualizations", f"{phase}_step_{step:06d}.png")
        plt.savefig(save_path, dpi=150)

        if self.writer:
            self.writer.add_figure(f'Visualization/{phase}', fig, step)
        plt.close(fig)

    # =========================================================================
    # --- [END] MODIFICATION FOR VISUALIZATION ---
    # =========================================================================

    def plot_training_history(self):
        """ Plot the training history including all four subplots. """
        if not self.history['train']['total']:
            print("Warning: No training history found to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sparse Matching Training History', fontsize=16)

        # --- Subplot [0, 0]: Total Loss ---
        axes[0, 0].plot(self.history['train']['total'], label='Train Loss')
        axes[0, 0].plot(self.history['val']['total'], label='Val Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Loss")
        all_total_losses = [l for l in self.history['train']['total'] + self.history['val']['total'] if np.isfinite(l)]
        if all_total_losses:
            min_loss = min(all_total_losses)
            max_loss = max(all_total_losses)
            padding = (max_loss - min_loss) * 0.1 if (max_loss - min_loss) > 1e-6 else 0.1
            axes[0, 0].set_ylim(bottom=min_loss - padding, top=max_loss + padding)

        # --- Subplot [0, 1]: Number of Valid Keypoints ---
        axes[0, 1].plot(self.history['train']['num_valid_keypoints'], label='Train Keypoints')
        axes[0, 1].plot(self.history['val']['num_valid_keypoints'], label='Val Keypoints')
        axes[0, 1].set_title('Number of Valid Keypoints')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_xlabel("Epochs")
        axes[0, 1].set_ylabel("Count")

        # --- Subplot [1, 0]: Mean Disparity ---
        axes[1, 0].plot(self.history['train']['mean_disparity'], label='Train Mean Disparity')
        axes[1, 0].plot(self.history['val']['mean_disparity'], label='Val Mean Disparity')
        axes[1, 0].set_title('Mean Disparity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_xlabel("Epochs")
        axes[1, 0].set_ylabel("Disparity (pixels)")

        # --- (Phase 2) Subplot [1, 1]: Weighted Loss Components ---
        weighted_smooth_train = [s * self.cfg.SMOOTHNESS_WEIGHT for s in self.history['train']['smoothness']]
        feature_train = [s * self.cfg.FEATURE_WEIGHT for s in self.history['train']['feature']]
        photo_train = [s * self.cfg.PHOTOMETRIC_WEIGHT for s in self.history['train']['photometric']]

        axes[1, 1].plot(feature_train, label=f'Feature Loss (w={self.cfg.FEATURE_WEIGHT:.2f})')
        axes[1, 1].plot(photo_train, label=f'Photometric Loss (w={self.cfg.PHOTOMETRIC_WEIGHT:.2f})')
        axes[1, 1].plot(weighted_smooth_train, label=f'Smoothness Loss (w={self.cfg.SMOOTHNESS_WEIGHT:.4f})')
        axes[1, 1].set_title('Weighted Loss Components (Train)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].set_ylabel("Weighted Loss Contribution")
        all_comp_losses = [l for l in feature_train + photo_train + weighted_smooth_train if np.isfinite(l)]
        if all_comp_losses:
            min_loss_comp = min(all_comp_losses)
            max_loss_comp = max(all_comp_losses)
            padding_comp = (max_loss_comp - min_loss_comp) * 0.1 if (max_loss - min_loss) > 1e-6 else 0.1
            axes[1, 1].set_ylim(bottom=min_loss_comp - padding_comp, top=max_loss_comp + padding_comp)

        # --- Final Adjustments and Saving ---
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.run_dir, "visualizations", "training_history.png")
        try:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Warning: Failed to save training history plot: {e}")
        plt.close(fig)

    # =========================================================================
    # --- [START] v19 JSON Log Fix ---
    # =========================================================================
    def update_log_file(self, epoch):
        log_data = {
            'config': asdict(self.cfg),  # This should be a simple dict
            'epoch': epoch,
            'history': self.history  # This is a dict of lists of floats/ints
        }
        try:
            with open(self.log_file, 'w') as f:
                # Use a more robust default handler to prevent circular refs
                def default_converter(o):
                    if isinstance(o, (np.float32, np.float64)):
                        return float(o)
                    if isinstance(o, (np.int32, np.int64)):
                        return int(o)
                    if isinstance(o, np.ndarray):
                        return o.tolist()  # Convert arrays to lists
                    # Raise an error for unhandled types
                    return f"Unserializable type: {o.__class__.__name__}"

                json.dump(log_data, f, indent=2, default=default_converter)
        except Exception as e:
            # Print a more informative warning
            print(f"Warning: Failed to write log file: {e}")
            if "Circular" in str(e):
                print("--> Hint: A non-serializable object (like a tensor) might be in the log data.")
    # =========================================================================
    # --- [END] v19 JSON Log Fix ---
    # =========================================================================


# =========================================================================
# --- [START] FINAL SCRIPT - TRAINING MODE ENABLED ---
# =========================================================================
if __name__ == "__main__":
    cfg = Config()

    # --- Auto-tuning DISABLED ---
    print(f"--- Auto-tuning DISABLED ---")

    if cfg.BATCH_SIZE == 0:
        print("[WARNING] Auto-tuned batch size is 0. Setting to 1.")
        cfg.BATCH_SIZE = 1

    trainer = Trainer(cfg)

    # --- Detector Check Logic (DISABLED) ---
    # print("\n" + "="*80)
    # print("--- Running Detector Check Mode (NOT Training) ---")
    # ... (rest of check logic is removed) ...
    # sys.exit(0)

    # --- [FINAL] TRAINING ENABLED ---
    print("\n" + "=" * 80)
    print("--- Detector check complete. PROCEEDING TO TRAINING. ---")
    print("=" * 80)
    trainer.train()
# =========================================================================
# --- [END] FINAL SCRIPT ---
# =================================_========================================