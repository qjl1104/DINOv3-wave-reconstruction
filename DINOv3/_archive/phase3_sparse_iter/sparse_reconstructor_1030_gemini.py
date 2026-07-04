# sparse_matching_stereo.py (v2 - Feature-Metric Loss)
# MODIFIED FOR DETECTOR CHECK v15 (Final Parameter Tune)
# This script will not train. It will load 1 batch,
# run the blob keypoint detector, save a visualization, and exit.
# FIX: Reads BOTH ROIs, calculates the minimum common dimensions, and CROPS
#      both images to that size instead of resizing.
# FIX v10: Uses robust thresholding.
# FIX v15: FINAL TUNE - Lowering minThreshold (50->30) and minArea (20->10)
#          to catch the weaker signals in the right image.

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

    # --- MODIFICATION v7: Set to 0 to auto-detect from calibration file ---
    IMAGE_HEIGHT: int = 0
    IMAGE_WIDTH: int = 0
    # --- END MODIFICATION v7 ---

    MASK_THRESHOLD: int = 30

    # Sparse matching settings
    MAX_KEYPOINTS: int = 512  # Maximum keypoints per image
    NMS_RADIUS: int = 3  # Non-maximum suppression radius (NO LONGER USED BY BLOB DETECTOR)
    PATCH_SIZE: int = 5
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
    PHOTOMETRIC_WEIGHT: float = 1.0
    SMOOTHNESS_WEIGHT: float = 0.0001
    USE_ADVANCED_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.8
    EARLY_STOPPING_PATIENCE: int = 25


def auto_tune_config(cfg: Config):
    # This function is now SKIPPED
    pass


# =========================================================================
# --- [START] REPLACEMENT OF SparseKeypointDetector (v15 - Final Tune) ---
# =========================================================================
class SparseKeypointDetector(nn.Module):
    """
    Detect sparse keypoints using OpenCV's SimpleBlobDetector.
    This is tuned to find circular AND ELLIPTICAL white markers.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_keypoints = cfg.MAX_KEYPOINTS

        # --- Setup Blob Detector ---
        params = cv2.SimpleBlobDetector_Params()

        # --- MODIFICATION v10: Use Thresholding instead of Color ---
        params.filterByColor = False  # <--- IMPORTANT

        # --- MODIFICATION v15: Lower threshold for right image ---
        params.minThreshold = 30  # Start looking for blobs at this brightness (Lowered from 50)
        params.maxThreshold = 255  # Stop looking
        params.thresholdStep = 10  # Check every 10 brightness steps
        # --- END MODIFICATION v15 ---

        # --- MODIFICATION v15: Lower minArea for right image ---
        params.filterByArea = True
        params.minArea = 10  # pixels (Lowered from 20)
        params.maxArea = 5000  # pixels (Keep high for close dots)
        # --- END MODIFICATION v15 ---

        # Filter by Circularity (Relaxed to allow ellipses)
        params.filterByCircularity = True
        params.minCircularity = 0.3  # Lowered from 0.6

        # Filter by Convexity (less important, but good)
        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Filter by Inertia (Relaxed to allow ellipses)
        params.filterByInertia = True
        params.minInertiaRatio = 0.1  # Lowered from 0.4

        # Create detector
        self.detector = cv2.SimpleBlobDetector_create(params)
        print(f"--- Initialized SimpleBlobDetector (v15 - Final Tune) ---")
        print(f"Thresholds: [{params.minThreshold}, {params.maxThreshold}], Step: {params.thresholdStep}")
        print(f"Filter by Area: {params.filterByArea}, Range: [{params.minArea}, {params.maxArea}]")
        print(f"Filter by Circularity: {params.filterByCircularity}, Min: {params.minCircularity}")
        print(f"Filter by Inertia Ratio: {params.filterByInertia}, Min: {params.minInertiaRatio}")
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
            img_masked_tensor = img_tensor * mask_tensor  # Should work now

            # Convert from (0.0 - 1.0) float to (0 - 255) uint8
            img_np = (img_masked_tensor.cpu().numpy() * 255).astype(np.uint8)

            # Detect blobs
            # Note: The mask is already applied, so detector runs on masked image
            cv_keypoints = self.detector.detect(img_np)

            if not cv_keypoints:
                # No keypoints found
                keypoints_list.append(torch.zeros(1, 2, device=device))
                scores_list.append(torch.zeros(1, device=device))
                continue

            # Extract coords and scores (use blob size as score)
            kp_coords = np.array([kp.pt for kp in cv_keypoints]).astype(np.float32)
            kp_scores = np.array([kp.size for kp in cv_keypoints]).astype(np.float32)

            # Convert back to tensor
            kp_tensor = torch.from_numpy(kp_coords).to(device)
            scores_tensor = torch.from_numpy(kp_scores).to(device)

            # Limit to max_keypoints (if detector finds too many)
            # We sort by size (score) descending
            if len(kp_tensor) > self.max_keypoints:
                sorted_indices = torch.argsort(scores_tensor, descending=True)
                kp_tensor = kp_tensor[sorted_indices[:self.max_keypoints]]
                scores_tensor = scores_tensor[sorted_indices[:self.max_keypoints]]

            keypoints_list.append(kp_tensor)
            scores_list.append(scores_tensor)

        # Pad to same length (identical logic as before)
        max_len = max(len(kp) for kp in keypoints_list)
        if max_len == 0: max_len = 1  # Handle case where all images are empty

        keypoints_padded = []
        scores_padded = []

        for kp, sc in zip(keypoints_list, scores_list):
            if len(kp) == 0:  # Handle empty tensor case explicitly
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

        keypoints = torch.stack(keypoints_padded, dim=0)  # (B, N, 2)
        scores = torch.stack(scores_padded, dim=0)  # (B, N)

        return keypoints, scores


# =========================================================================
# --- [END] REPLACEMENT OF SparseKeypointDetector ---
# =========================================================================


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


# --- 4. NEW: Feature-Metric Loss ---
class FeatureMetricLoss(nn.Module):
    """
    Computes loss based on DINOv3 feature similarity, not pixel intensity.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, desc_left, desc_right, match_scores, keypoints_left, disparity, scores_left, constraint_mask):
        B, N_l, C = desc_left.shape
        device = desc_left.device
        match_probs = F.softmax(match_scores * self.cfg.MATCHING_TEMPERATURE, dim=2)
        match_probs = torch.nan_to_num(match_probs, nan=0.0)
        desc_right_weighted = torch.bmm(match_probs, desc_right)
        eps = 1e-8
        desc_left_norm = F.normalize(desc_left, dim=2, eps=eps)
        desc_right_weighted_norm = F.normalize(desc_right_weighted, dim=2, eps=eps)
        cosine_sim = (desc_left_norm * desc_right_weighted_norm).sum(dim=2)
        feature_loss_per_kp = 1.0 - cosine_sim
        detection_mask = (scores_left > 0.1)  # Use blob size as score, so 0.1 is fine
        matchable_mask = torch.any(constraint_mask, dim=2)
        final_valid_mask = detection_mask & matchable_mask
        masked_loss = feature_loss_per_kp * final_valid_mask
        num_valid = torch.sum(final_valid_mask)
        feature_loss = torch.tensor(0.0, device=device)
        if num_valid > 0:
            feature_loss = masked_loss.sum() / num_valid
        smooth_loss = self._compute_sparse_smoothness(keypoints_left, disparity, scores_left)
        return feature_loss, smooth_loss

    def _compute_sparse_smoothness(self, keypoints, disparity, scores):
        B, N, _ = keypoints.shape
        smooth_loss = 0.0
        valid_pairs = 0
        for b in range(B):
            kp = keypoints[b]
            disp = disparity[b]
            sc = scores[b]
            valid_mask = sc > 0.1  # Use blob size as score
            kp_valid = kp[valid_mask]
            disp_valid = disp[valid_mask]
            if len(kp_valid) < 2:
                continue
            dist = torch.cdist(kp_valid, kp_valid)
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

            # --- MODIFICATION v9: Auto-set config resolution to MINIMUM of ROIs ---
            _x_l, _y_l, w_l, h_l = self.roi_left
            _x_r, _y_r, w_r, h_r = self.roi_right

            if self.cfg.IMAGE_WIDTH == 0 or self.cfg.IMAGE_HEIGHT == 0:
                print(f"[Dataset] Left ROI: {w_l}x{h_l}, Right ROI: {w_r}x{h_r}")
                target_w = min(w_l, w_r)
                target_h = min(h_l, h_r)
                print(f"[Dataset] Setting target config resolution to MINIMUM common ROI: {target_w}x{target_h}")
                self.cfg.IMAGE_WIDTH = target_w
                self.cfg.IMAGE_HEIGHT = target_h
            # --- END MODIFICATION v9 ---

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
            left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

            # Crop to respective ROIs first
            x, y, w, h = self.roi_left
            left_rect = left_rect[y:y + h, x:x + w]
            x, y, w, h = self.roi_right
            right_rect = right_rect[y:y + h, x:x + w]

            # --- MODIFICATION v9: CROP both images to the minimum common size ---
            # --- This replaces the buggy cv2.resize logic ---
            left_img = left_rect[0:self.cfg.IMAGE_HEIGHT, 0:self.cfg.IMAGE_WIDTH]
            right_img = right_rect[0:self.cfg.IMAGE_HEIGHT, 0:self.cfg.IMAGE_WIDTH]
            # --- END MODIFICATION v9 ---

            # Sanity check shapes
            if left_img.shape[0] != self.cfg.IMAGE_HEIGHT or left_img.shape[1] != self.cfg.IMAGE_WIDTH:
                print(
                    f"[ERROR] Left image shape mismatch after crop! Got {left_img.shape}, expected ({self.cfg.IMAGE_HEIGHT},{self.cfg.IMAGE_WIDTH})")
                return None
            if right_img.shape[0] != self.cfg.IMAGE_HEIGHT or right_img.shape[1] != self.cfg.IMAGE_WIDTH:
                print(
                    f"[ERROR] Right image shape mismatch after crop! Got {right_img.shape}, expected ({self.cfg.IMAGE_HEIGHT},{self.cfg.IMAGE_WIDTH})")
                return None

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
            _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
            left_gray = torch.from_numpy(left_img).float().unsqueeze(0) / 255.0
            right_gray = torch.from_numpy(right_img).float().unsqueeze(0) / 255.0
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
            import traceback
            traceback.print_exc()
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
        print(f"Using NEW Feature-Metric Loss (Cosine Similarity)")

    def forward(self, left_gray, right_gray, left_rgb, right_rgb, mask):
        B, _, H, W = left_gray.shape
        kp_left, scores_left = self.keypoint_detector(left_gray, mask)
        kp_right, scores_right = self.keypoint_detector(right_gray, mask)
        desc_left = self.feature_extractor(left_rgb, kp_left)
        desc_right = self.feature_extractor(right_rgb, kp_right)
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


# --- 7. Evaluation Metrics (独立类，不要嵌套!) ---
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

        # --- MODIFICATION v9: Print the final chosen resolution ---
        print(f"--- Using FINAL resolution (min common ROI): {self.cfg.IMAGE_WIDTH}x{self.cfg.IMAGE_HEIGHT} ---")
        # --- END MODIFICATION v9 ---

        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_fn, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=0, pin_memory=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tensorboard")) if SummaryWriter else None
        self.model = SparseMatchingStereoModel(cfg).to(self.device)
        self.loss_fn = FeatureMetricLoss(cfg)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=cfg.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-7
        )
        self.evaluator = EvaluationMetrics()
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION)
        self.step = 0
        self.log_file = os.path.join(self.run_dir, "logs", "training_log.json")
        self.loss_keys = ['total', 'feature', 'smoothness']
        self.metric_keys = ['mean_disparity', 'std_disparity', 'num_valid_keypoints']
        self.history = {
            'train': {k: [] for k in self.loss_keys + self.metric_keys},
            'val': {k: [] for k in self.loss_keys + self.metric_keys}
        }

    def train(self):
        print("\n--- Starting Sparse Matching Training (v2: Feature-Metric Loss) ---")
        print(f"Max keypoints: {self.cfg.MAX_KEYPOINTS}, Loss Temp: {self.cfg.MATCHING_TEMPERATURE}")
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

    # =========================================================================
    # --- [START] PADDING FIX v8 ---
    # =========================================================================
    def _pad_inputs(self, *tensors):
        """Pad EACH tensor INDIVIDUALLY to the target size needed for DINOv3 patch size"""
        patch_size = 16
        padded_tensors = []
        target_h, target_w = -1, -1  # Calculate target based on the first tensor

        for i, x in enumerate(tensors):
            if x is None:  # Handle potential None tensors if dataset had issues
                padded_tensors.append(None)
                continue

            _, _, h, w = x.shape

            if i == 0:  # Calculate target dimensions from the first tensor
                pad_h = (patch_size - h % patch_size) % patch_size
                pad_w = (patch_size - w % patch_size) % patch_size
                target_h = h + pad_h
                target_w = w + pad_w
                # print(f"Padding target (H, W): ({target_h}, {target_w})") # Debug print

            # Calculate padding needed *for this specific tensor* to reach target_h, target_w
            current_pad_h = target_h - h
            current_pad_w = target_w - w

            # Ensure padding is non-negative (shouldn't happen if target is calculated correctly)
            current_pad_h = max(0, current_pad_h)
            current_pad_w = max(0, current_pad_w)

            if current_pad_h > 0 or current_pad_w > 0:
                # Pad only right and bottom: (pad_left, pad_right, pad_top, pad_bottom)
                padding = (0, current_pad_w, 0, current_pad_h)
                try:
                    padded_x = F.pad(x, padding, mode='constant', value=0)
                    # Sanity check shape after padding
                    if padded_x.shape[2] != target_h or padded_x.shape[3] != target_w:
                        print(
                            f"[ERROR] Padding failed for tensor {i}. Input ({h},{w}), Target ({target_h},{target_w}), Got {padded_x.shape[2:]}")
                        padded_tensors.append(x)
                    else:
                        padded_tensors.append(padded_x)
                except Exception as e:
                    print(f"[ERROR] Exception during F.pad for tensor {i}: {e}")
                    padded_tensors.append(x)  # Fallback on error

            else:
                padded_tensors.append(x)  # No padding needed

        # --- MODIFICATION v9: Removed the confusing shape warning ---
        return padded_tensors

    # =========================================================================
    # --- [END] PADDING FIX v8 ---
    # =========================================================================

    def _run_epoch(self, epoch, is_training):
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [{'Train' if is_training else 'Val'}]")
        epoch_results = {k: 0.0 for k in self.loss_keys + self.metric_keys}
        for data in pbar:
            if data is None:
                continue
            left_gray = data['left_gray'].to(self.device)
            right_gray = data['right_gray'].to(self.device)
            left_rgb = data['left_rgb'].to(self.device)
            right_rgb = data['right_rgb'].to(self.device)
            mask = data['mask'].to(self.device)

            # --- Call the NEW padding function ---
            padded_inputs = self._pad_inputs(left_gray, right_gray, left_rgb, right_rgb, mask)
            left_gray, right_gray, left_rgb, right_rgb, mask = padded_inputs
            # --- Check if padding returned valid tensors ---
            if None in padded_inputs:
                print(f"[ERROR] Padding returned None for some inputs in _run_epoch. Skipping batch.")
                continue

            with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION):
                outputs = self.model(left_gray, right_gray, left_rgb, right_rgb, mask)
                feature_loss, smooth_loss = self.loss_fn(
                    outputs['descriptors_left'],
                    outputs['descriptors_right'],
                    outputs['match_scores'],
                    outputs['keypoints_left'],
                    outputs['disparity'],
                    outputs['scores_left'],
                    outputs['constraint_mask']
                )
                total_loss = (self.cfg.PHOTOMETRIC_WEIGHT * feature_loss +
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
            metrics = self.evaluator.compute_sparse_metrics(
                outputs['disparity'],
                outputs['keypoints_left'],
                outputs['scores_left']
            )
            epoch_results['total'] += total_loss.item() if torch.isfinite(total_loss) else 0.0
            epoch_results['feature'] += feature_loss.item() if torch.isfinite(feature_loss) else 0.0
            epoch_results['smoothness'] += smooth_loss.item() if torch.isfinite(smooth_loss) else 0.0
            for k in self.metric_keys:
                if k in metrics:
                    epoch_results[k] += metrics[k]
            pbar.set_postfix({
                'loss': total_loss.item(),
                'feat_loss': feature_loss.item(),
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
        if num_batches > 0:
            return {k: v / num_batches for k, v in epoch_results.items()}
        else:
            print("Warning: Loader was empty, returning zeroed results.")
            return epoch_results

    def _log_epoch_results(self, phase, epoch, results):
        for k, v in results.items():
            self.history[phase][k].append(v)
            if self.writer:
                metric_type = 'Loss' if k in self.loss_keys else 'Metrics'
                if k == 'smoothness':
                    self.writer.add_scalar(f"Loss/{phase}_smoothness_unweighted", v, epoch)
                else:
                    self.writer.add_scalar(f"{metric_type}/{phase}_{k}", v, epoch)

    # =========================================================================
    # --- [START] MODIFICATION FOR DETECTOR CHECK ---
    # =========================================================================
    def visualize(self, data, outputs, step, phase):
        """
        MODIFIED: Visualize sparse keypoints on LEFT and RIGHT images
        """
        # --- Get Left Image Data ---
        # --- Use ORIGINAL data for visualization, not padded ---
        left_gray_orig = data['left_gray'][0, 0].cpu().numpy()  # (H_orig, W_orig)
        kp_left = outputs['keypoints_left'][0].cpu().detach().numpy()  # (N, 2) [These coords are in PADDED space]
        scores_left = outputs['scores_left'][0].cpu().detach().numpy()  # (N,)

        # --- Get Right Image Data ---
        right_gray_orig = data['right_gray'][0, 0].cpu().numpy()  # (H_orig, W_orig)
        kp_right = outputs['keypoints_right'][0].cpu().detach().numpy()  # (N, 2) [These coords are in PADDED space]
        scores_right = outputs['scores_right'][0].cpu().detach().numpy()  # (N,)

        # --- Filter valid Left keypoints ---
        valid_mask_left = scores_left > 0.1
        kp_left_valid = kp_left[valid_mask_left]

        # --- Filter valid Right keypoints ---
        valid_mask_right = scores_right > 0.1
        kp_right_valid = kp_right[valid_mask_right]

        # --- Create Plot ---
        # --- Increase figure size for high-res images ---
        h, w = left_gray_orig.shape  # Use original shape for aspect ratio
        fig_width = 15
        fig_height = (h / w) * fig_width * 0.5 if w > 0 else fig_width * 0.5 * (3 / 4)  # Aspect ratio calc
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height + 1))  # Add a bit extra height

        # --- FIX v14: SyntaxError fontsize: -> fontsize= ---
        fig.suptitle(f'DETECTOR CHECK - Step: {step} ({phase})', fontsize=16)

        # --- Left: keypoints on image ---
        axes[0].imshow(left_gray_orig, cmap='gray')  # Show original image
        if len(kp_left_valid) > 0:
            # Keypoint coordinates are relative to the PADDED image, but we display the ORIGINAL.
            # Since padding is only on right/bottom, the coordinates should still be correct relative to top-left.
            axes[0].scatter(kp_left_valid[:, 0], kp_left_valid[:, 1], c='red', s=20, alpha=0.6)
        axes[0].set_title(f"Detected Left Keypoints ({len(kp_left_valid)})")
        axes[0].axis('off')

        # --- Right: keypoints on image ---
        axes[1].imshow(right_gray_orig, cmap='gray')  # Show original image
        if len(kp_right_valid) > 0:
            axes[1].scatter(kp_right_valid[:, 0], kp_right_valid[:, 1], c='cyan', s=20, alpha=0.6)
        axes[1].set_title(f"Detected Right Keypoints ({len(kp_right_valid)})")
        axes[1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.run_dir, "visualizations", f"{phase}_step_{step:06d}.png")

        # --- Save with higher DPI ---
        plt.savefig(save_path, dpi=150)  # Increased from 100

        if self.writer:
            self.writer.add_figure(f'Visualization/{phase}', fig, step)
        plt.close(fig)

    # =========================================================================
    # --- [END] MODIFICATION FOR DETECTOR CHECK ---
    # =========================================================================

    def plot_training_history(self):
        """ Plot the training history including all four subplots. """
        if not self.history['train']['total']:
            print("Warning: No training history found to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        # --- FIX v14: SyntaxError fontsize: -> fontsize= ---
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

        # --- Subplot [1, 1]: Weighted Loss Components ---
        weighted_smooth_train = [s * self.cfg.SMOOTHNESS_WEIGHT for s in self.history['train']['smoothness']]
        feature_train = self.history['train']['feature']
        axes[1, 1].plot(feature_train, label=f'Feature Loss (Weight: {self.cfg.PHOTOMETRIC_WEIGHT:.4f})')
        axes[1, 1].plot(weighted_smooth_train, label=f'Smoothness Loss (Weight: {self.cfg.SMOOTHNESS_WEIGHT:.4f})')
        axes[1, 1].set_title('Weighted Loss Components (Train)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_xlabel("Epochs")
        axes[1, 1].set_ylabel("Weighted Loss Contribution")
        all_comp_losses = [l for l in feature_train + weighted_smooth_train if np.isfinite(l)]
        if all_comp_losses:
            min_loss_comp = min(all_comp_losses)
            max_loss_comp = max(all_comp_losses)
            padding_comp = (max_loss_comp - min_loss_comp) * 0.1 if (max_loss_comp - min_loss_comp) > 1e-6 else 0.1
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

    def update_log_file(self, epoch):
        log_data = {'config': asdict(self.cfg), 'epoch': epoch, 'history': self.history}
        try:
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2,
                          default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else str(
                              o) if isinstance(o, np.ndarray) else o)
        except Exception as e:
            print(f"Warning: Failed to write log file: {e}")


# =========================================================================
# --- [START] MODIFICATION FOR DETECTOR CHECK ---
# =========================================================================
if __name__ == "__main__":
    cfg = Config()

    # --- Auto-tuning DISABLED ---
    print(f"--- Auto-tuning DISABLED ---")
    # Resolution will be set by the Dataset loader based on CALIBRATION_FILE ROI

    if cfg.BATCH_SIZE == 0:
        print("[WARNING] Auto-tuned batch size is 0. Setting to 1.")
        cfg.BATCH_SIZE = 1

    trainer = Trainer(cfg)

    # --- Detector Check Logic ---
    print("\n" + "=" * 80)
    print("--- Running Detector Check Mode (NOT Training) ---")
    print("=" * 80)

    # 1. Get one batch of data
    try:
        print("Loading one batch from train_loader...")
        data = next(iter(trainer.train_loader))
        if data is None:
            print("[ERROR] Failed to load the first batch (data is None). Check dataset path and collate_fn.")
            sys.exit(1)
        print("Successfully loaded one batch of data.")
    except StopIteration:
        print("[ERROR] train_loader is empty. Check your dataset path and validation split.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load data from train_loader: {e}")
        sys.exit(1)

    # 2. Move to device
    try:
        left_gray = data['left_gray'].to(trainer.device)
        right_gray = data['right_gray'].to(trainer.device)
        left_rgb = data['left_rgb'].to(trainer.device)
        right_rgb = data['right_rgb'].to(trainer.device)
        mask = data['mask'].to(trainer.device)
    except Exception as e:
        print(f"[ERROR] Failed to move data to device: {e}")
        sys.exit(1)

    # 3. Pad inputs using the FIXED function
    print(f"Padding inputs... (Left Image shape: {left_gray.shape}, Right Image shape: {right_gray.shape})")
    padded_inputs = trainer._pad_inputs(left_gray, right_gray, left_rgb, right_rgb, mask)
    left_gray, right_gray, left_rgb, right_rgb, mask = padded_inputs
    if None in padded_inputs:
        print(f"[ERROR] Padding returned None. Cannot proceed.")
        sys.exit(1)
    print(f"Padded inputs. (New shape: {left_gray.shape})")

    # 4. Run forward pass (model is part of trainer)
    trainer.model.eval()  # Set to evaluation mode
    with torch.no_grad():
        print("Running model forward pass (this runs the detector)...")
        try:
            outputs = trainer.model(left_gray, right_gray, left_rgb, right_rgb, mask)
        except Exception as e:
            print(f"\n[ERROR] Exception during model forward pass: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    print("Forward pass complete. Generating visualization...")

    # 5. Call the (modified) visualize function
    try:
        # Pass ORIGINAL (unpadded) data for visualization, but outputs from padded run
        original_data_vis = {
            'left_gray': data['left_gray'],
            'right_gray': data['right_gray']
        }
        trainer.visualize(original_data_vis, outputs, 0, "check")

        # --- FIX v13b: Use trainer.run_dir instead of self.run_dir ---
        save_path = os.path.join(trainer.run_dir, "visualizations", "check_step_000000.png")
        # --- END FIX v13b ---

        print("\n" + "=" * 80)
        print(f"--- [SUCCESS] ---")
        print(f"Detector check visualization saved to:")
        print(f"{save_path}")
        print("=" * 80)
        print("\nACTION: Please open this image and check the keypoints.")

    except Exception as e:
        print(f"\n[ERROR] Failed to generate visualization: {e}")
        import traceback

        traceback.print_exc()

    # 6. Exit
    print("--- Detector Check Complete ---")
    sys.exit(0)

    # trainer.train() # <--- Original training call is SKIPPED
# =========================================================================
# --- [END] MODIFICATION FOR DETECTOR CHECK ---
# =================================_========================================

