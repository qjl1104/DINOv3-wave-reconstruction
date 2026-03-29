# sparse_matching_stereo.py (v2 - Feature-Metric Loss)
# ... (v2.1 到 v2.11 的注释) ...
#
# v2.12 (Mask Fix):
# - (已废弃) 移除 Dataset 中的 MASK_THRESHOLD...
#
# v2.13 (Detector Fix):
# - v2.12 的修复不完整。真正的问题在 SparseKeypointDetector 内部。
# - `values > 0.1` 的硬编码阈值在高分辨率下仍然过高。
# - 解决方案：将此阈值更改为 `values > 0.0`。
#   这确保我们总是获取 top-k *最亮* 的点，而不是 top-k *足够亮* 的点。
#

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

# 尝试导入 pynvml 以便更准确地检查 VRAM
try:
    import pynvml
except ImportError:
    pynvml = None
    # --- MODIFICATION: 将打印移至 main 块 ---
    # print("[信息] pynvml 未安装。 无法准确显示可用显存。")
    # print("       (可选) 尝试运行: pip install nvidia-ml-py")
    # --- END MODIFICATION ---

# Assume the script is run from the DINOv3 directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # More robust way to find project root
# PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3" # Keep original if needed
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


@dataclass
class Config:
    """Configuration for sparse matching"""
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")
    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs_sparse")
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")  # Assumes model is in project root

    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 100

    # --- v2.11 MODIFICATION: 16GB Hires ---
    # 手动设置分辨率。为 16GB VRAM 设置一个合理的起始点。
    # 必须是 16 的倍数 (DINOv3 patch size)
    # IMAGE_HEIGHT: int = 640  # 16GB 起始点 (16:10 比例)
    # IMAGE_WIDTH: int = 1024  # 16GB 起始点
    # --- END MODIFICATION ---

    # =========================================================================
    # --- [START] MODIFICATION: 恢复 Baseline 的自动分辨率检测 ---
    # =========================================================================
    # (从 ...1030_gemini.py 移植而来)
    # 设置为 0, Dataset 将从标定文件的 ROI 中自动检测
    IMAGE_HEIGHT: int = 0
    IMAGE_WIDTH: int = 0
    # =========================================================================
    # --- [END] MODIFICATION ---
    # =========================================================================

    MASK_THRESHOLD: int = 30  # (v2.12/v2.13 已在 Dataset 中禁用)

    # --- v2.11 MODIFICATION: Dilation ---
    USE_DILATION: bool = False  # 禁用膨胀 (DINOv3 不需要)
    DILATION_KERNEL_SIZE: int = 5  # (未使用)
    # --- END MODIFICATION ---

    # Sparse matching settings
    # --- MODIFICATION: 恢复为原始的稀疏设置 ---
    MAX_KEYPOINTS: int = 512  # (Original: 512) 恢复为 512 (符合 400-500 的要求)
    NMS_RADIUS: int = 3  # (Original: 3) 恢复为 3
    # --- END MODIFICATION ---
    PATCH_SIZE: int = 5  # (DEPRECATED by FeatureMetricLoss) Patch size for photometric loss (5x5)

    # Matching network settings
    FEATURE_DIM: int = 768  # DINOv3 feature dimension
    # --- v2.6 MODIFICATION: Increase Attention Layers ---
    NUM_ATTENTION_LAYERS: int = 6  # Number of attention layers in matcher (Increased from 4)
    # --- END MODIFICATION ---
    NUM_HEADS: int = 8  # Number of attention heads
    DISPARITY_CONSTRAINT_Y_THRESHOLD: int = 2  # Y-pixel tolerance for epipolar constraint
    MATCHING_TEMPERATURE: float = 10.0  # Temperature for softmax in matcher and loss

    # Training settings
    BATCH_SIZE: int = 1
    # --- v2.6 MODIFICATION: Revert Learning Rate ---
    LEARNING_RATE: float = 1e-4  # AdamW learning rate (Reverted from 1e-5)
    # --- END MODIFICATION ---
    NUM_EPOCHS: int = 100  # Maximum training epochs
    VALIDATION_SPLIT: float = 0.1  # Fraction of data for validation
    GRADIENT_CLIP_VAL: float = 1.0  # Max gradient norm for clipping

    # --- v2.11 Hires Note ---
    # 为 16GB VRAM 设置。如果 OOM, 保持 2。
    # 如果显存充足 (例如 < 12GB 占用), 尝试改回 1。
    # --- MODIFICATION: 恢复为原始的累积设置 ---
    # 关键点已恢复为 512，我们不再需要高累积
    GRADIENT_ACCUMULATION_STEPS: int = 2  # (Original: 2) 恢复原始值以提速
    # --- END MODIFICATION ---

    # Performance and Loss settings
    USE_MIXED_PRECISION: bool = True  # Use AMP for faster training
    PHOTOMETRIC_WEIGHT: float = 1.0  # Weight for the FEATURE loss (Cosine Similarity)

    # --- v2.7 MODIFICATION: Adjust Smoothness Weight ---
    # SMOOTHNESS_WEIGHT: float = 0.001  # Weight for the smoothness regularizer (Increased from 0.0001)
    # =========================================================================
    # --- [START] MODIFICATION: 提高平滑权重以帮助收敛 ---
    # =========================================================================
    SMOOTHNESS_WEIGHT: float = 0.01  # 提高权重 (原为 0.001), 帮助模型跳出局部最优
    # =========================================================================
    # --- [END] MODIFICATION ---
    # =========================================================================
    # --- END MODIFICATION ---

    # Data Augmentation settings
    USE_ADVANCED_AUGMENTATION: bool = True  # Enable brightness/contrast augmentation
    AUGMENTATION_PROBABILITY: float = 0.8  # Probability of applying augmentation

    # Early Stopping settings
    EARLY_STOPPING_PATIENCE: int = 25  # Stop if val loss doesn't improve for N epochs


def auto_tune_config(cfg: Config):
    """ (已禁用) Automatically adjust batch size and resolution based on available GPU memory. """
    print("--- Probing GPU for auto-tuning ---")
    try:
        # Query free GPU memory using nvidia-smi
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      stderr=subprocess.DEVNULL).decode("utf-8").strip()
        free_mem_mb = int(out)
    except Exception as e:
        print(f"[WARNING] Could not probe GPU: {e}. Using default settings.")
        free_mem_mb = None

    patch_size = 16  # DINOv3 patch size
    if not free_mem_mb:
        return cfg

    print(f"Available GPU Memory: {free_mem_mb} MB")

    # Adjust parameters based on memory (heuristics, may need tuning)
    if free_mem_mb < 8000:  # Low memory
        cfg.BATCH_SIZE, cfg.MAX_KEYPOINTS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 256, 4, 0.7
    elif free_mem_mb < 12000:  # Medium memory (11GB)
        cfg.BATCH_SIZE, cfg.MAX_KEYPOINTS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 512, 2, 0.8
    else:  # High memory (16GB+)
        cfg.BATCH_SIZE, cfg.MAX_KEYPOINTS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 512, 1, 1.0

    # Adjust image dimensions to be multiples of patch size
    cfg.IMAGE_WIDTH = int((cfg.IMAGE_WIDTH * scale) // patch_size) * patch_size
    cfg.IMAGE_HEIGHT = int((cfg.IMAGE_HEIGHT * scale) // patch_size) * patch_size

    print(f"  Auto-tuned -> Resolution: {cfg.IMAGE_WIDTH}x{cfg.IMAGE_HEIGHT}, Batch: {cfg.BATCH_SIZE}, "
          f"Max Keypoints: {cfg.MAX_KEYPOINTS}, Grad Accum: {cfg.GRADIENT_ACCUMULATION_STEPS}")
    return cfg


# --- 1. Sparse Keypoint Detector ---
# =========================================================================
# --- [START] MODIFICATION: 恢复为 Baseline 的斑点检测器 ---
# =========================================================================
class SparseKeypointDetector(nn.Module):
    """
    Detect sparse keypoints using OpenCV's SimpleBlobDetector.
    This is tuned to find circular AND ELLIPTICAL white markers.
    (从 ...1030_gemini.py 移植而来)
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
        print(f"--- Initialized SimpleBlobDetector (移植自 v15 Baseline) ---")
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
            mask: (B, 1, H, W) - valid region mask (来自 cv2.threshold)
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
            # 我们假设 mask 是一个二值掩码 (0.0 or 1.0)
            img_masked_tensor = img_tensor * mask_tensor

            # Convert from (0.0 - 1.0) float to (0 - 255) uint8
            img_np = (img_masked_tensor.cpu().numpy() * 255).astype(np.uint8)

            # --- 关键: 我们在 *已经* 阈值化过的掩码图像上运行检测 ---
            # --- 注意: SimpleBlobDetector 仍会应用它 *自己的* 内部阈值 ---
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
        max_len = max(len(kp) for kp in keypoints_list) if keypoints_list else 0
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
                # Truncate if we have more than max_len
                kp_pad, sc_pad = kp[:max_len], sc[:max_len]
            keypoints_padded.append(kp_pad)
            scores_padded.append(sc_pad)

        keypoints = torch.stack(keypoints_padded, dim=0)  # (B, N, 2)
        scores = torch.stack(scores_padded, dim=0)  # (B, N)

        return keypoints, scores


# =========================================================================
# --- [END] MODIFICATION: 恢复为 Baseline 的斑点检测器 ---
# =========================================================================


# --- 2. DINOv3 Feature Extractor ---
class DINOv3FeatureExtractor(nn.Module):
    """Extracts DINOv3 features at specified keypoint locations using grid_sample."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Load the pre-trained DINOv3 model
        self.dino = self._load_dino_model()

        # Freeze DINOv3 parameters - it acts only as a feature extractor
        for p in self.dino.parameters():
            p.requires_grad = False

        # Store necessary config values from the loaded DINO model
        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        # Number of register tokens (specific to some DINO models)
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

    def _load_dino_model(self):
        """Loads the DINOv3 model from the specified local path."""
        try:
            # Attempt to load from local files only to ensure the correct version
            model_path = self.cfg.DINO_LOCAL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"DINO model directory not found at: {model_path}")
            print(f"Loading DINO model from: {model_path}")
            return AutoModel.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            # Exit if loading fails, as it's critical
            print(f"[FATAL] Failed loading local DINOv3 model from '{self.cfg.DINO_LOCAL_PATH}': {e}")
            sys.exit(1)

    def get_feature_map(self, image):
        """
        Extracts the spatial feature map from DINOv3's last layer.

        Args:
            image: (B, 3, H, W) - Input RGB image tensor.

        Returns:
            feat_map: (B, C, H_feat, W_feat) - Spatial feature map, where
                      C = feature_dim, H_feat = H // patch_size, W_feat = W // patch_size.
        """
        with torch.no_grad():  # Ensure no gradients are computed for DINO
            # Get the output of the last hidden state from DINO
            outputs = self.dino(image)
            if not hasattr(outputs, 'last_hidden_state'):
                print("[ERROR] DINO model output does not have 'last_hidden_state'. Check model compatibility.")
                # Fallback or error handling
                if hasattr(outputs, 'pooler_output'):
                    print("[WARNING] Using pooler_output instead, spatial information lost!")
                    return outputs.pooler_output.unsqueeze(-1).unsqueeze(-1)
                else:
                    print("[FATAL] Cannot extract features from DINO output.")
                    sys.exit(1)

            features = outputs.last_hidden_state  # Shape (B, num_tokens, C)

        b, _, h, w = image.shape
        # Calculate the starting index of patch tokens (skip CLS and register tokens)
        start_idx = 1 + self.num_register_tokens
        # Extract only the patch tokens
        # Add check for sufficient tokens
        if features.shape[1] <= start_idx:
            print(
                f"[ERROR] DINO output has insufficient tokens ({features.shape[1]}) to extract patch tokens (start_idx={start_idx}). Check image size and patch size.")
            return torch.zeros((b, self.feature_dim, h // self.patch_size, w // self.patch_size), device=image.device)

        patch_tokens = features[:, start_idx:, :]  # Shape (B, num_patches, C)
        # Calculate feature map dimensions
        h_feat, w_feat = h // self.patch_size, w // self.patch_size

        # Check if the number of patch tokens matches expected H_feat * W_feat
        expected_num_patches = h_feat * w_feat
        if patch_tokens.shape[1] != expected_num_patches:
            print(
                f"[WARNING] Number of patch tokens ({patch_tokens.shape[1]}) does not match expected ({expected_num_patches}). Check image padding or DINO model behavior.")
            if patch_tokens.shape[1] > expected_num_patches:
                patch_tokens = patch_tokens[:, :expected_num_patches, :]

        # Reshape patch tokens into a spatial feature map
        try:
            feat_map = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, h_feat, w_feat)
        except RuntimeError as e:
            print(
                f"[ERROR] Failed to reshape patch tokens. Patch token count: {patch_tokens.shape[1]}, Expected: {h_feat * w_feat}. Error: {e}")
            return torch.zeros((b, self.feature_dim, h_feat, w_feat), device=image.device)

        return feat_map

    def forward(self, image, keypoints):
        """
        Extracts DINOv3 features at the given keypoint locations.

        Args:
            image: (B, 3, H, W) - Input RGB image tensor.
            keypoints: (B, N, 2) - Keypoint coordinates (x, y) in image space.

        Returns:
            descriptors: (B, N, C) - Feature descriptors sampled at keypoint locations.
        """
        B, N, _ = keypoints.shape
        # Get the dense feature map from DINO
        feat_map = self.get_feature_map(image)  # (B, C, H_feat, W_feat)

        _, C, H_feat, W_feat = feat_map.shape
        _, _, H_img, W_img = image.shape

        # Handle potential zero dimensions
        if W_img <= 1 or H_img <= 1:
            print(
                f"[WARNING] Invalid image dimensions for normalization: W={W_img}, H={H_img}. Returning zero descriptors.")
            return torch.zeros((B, N, C), device=image.device)

        # --- Feature Sampling using grid_sample ---
        # Normalize keypoint coordinates from image space [0, W-1], [0, H-1]
        # to grid_sample's expected range [-1, 1].
        grid = keypoints.clone()
        grid[..., 0] = 2 * (grid[..., 0] / (W_img - 1)) - 1  # Normalize X
        grid[..., 1] = 2 * (grid[..., 1] / (H_img - 1)) - 1  # Normalize Y
        # grid_sample expects coordinates in (x, y) order.
        # Add an extra dimension for grid_sample: (B, N, 1, 2)
        grid = grid.unsqueeze(2)

        # Sample features from the map using bilinear interpolation
        try:
            descriptors_sampled = F.grid_sample(
                feat_map, grid,
                mode='bilinear',  # Use bilinear interpolation
                align_corners=True,  # Consistent coordinate mapping
                padding_mode='border'  # How to handle points outside the map
            )
        except Exception as e:
            print(f"[ERROR] F.grid_sample failed: {e}. Returning zero descriptors.")
            return torch.zeros((B, N, C), device=image.device)

        # Reshape the output to (B, N, C)
        descriptors = descriptors_sampled.squeeze(3).permute(0, 2, 1)

        return descriptors


# --- 3. Attention-based Sparse Matching Network ---
class PositionalEncoding(nn.Module):
    """Learned 2D positional encoding for keypoints."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # A simple linear layer projects normalized (x, y) coordinates to the feature dimension
        self.proj = nn.Linear(2, dim)

    def forward(self, positions, image_size):
        """
        Generates positional encodings for keypoint positions.

        Args:
            positions: (B, N, 2) - Keypoint coordinates (x, y) in image space.
            image_size: (H, W) - Tuple representing image dimensions.

        Returns:
            encoding: (B, N, dim) - Positional encoding vectors.
        """
        H, W = image_size
        # Handle potential zero dimensions
        if W <= 1 or H <= 1:
            print(f"[WARNING] Invalid image size for PositionalEncoding: W={W}, H={H}. Returning zero encoding.")
            return torch.zeros((positions.shape[0], positions.shape[1], self.dim),
                               device=positions.device)  # Match shape B, N, dim

        # Normalize coordinates to the range [0, 1]
        pos_normalized = positions.clone()
        pos_normalized[..., 0] = pos_normalized[..., 0] / (W - 1)  # Normalize X
        pos_normalized[..., 1] = pos_normalized[..., 1] / (H - 1)  # Normalize Y
        # Project normalized coordinates to the target dimension
        return self.proj(pos_normalized)


class SelfAttentionLayer(nn.Module):
    """Standard Transformer self-attention block with pre-layer normalization."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # Layer Normalization 1 (applied before attention)
        self.norm1 = nn.LayerNorm(dim)
        # Layer Normalization 2 (applied before FFN)
        self.norm2 = nn.LayerNorm(dim)
        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),  # Expand dimension
            nn.ReLU(),  # Activation
            nn.Linear(dim * 2, dim)  # Project back to original dimension
        )

    def forward(self, features, pos_enc):
        """
        Applies self-attention to features combined with positional encoding.

        Args:
            features: (B, N, dim) - Input feature vectors.
            pos_enc: (B, N, dim) - Positional encoding vectors.

        Returns:
            features: (B, N, dim) - Output features after attention and FFN.
        """
        # --- Self-Attention with Pre-LayerNorm and Residual Connection ---
        features_pos = features + pos_enc  # Add positional encoding
        qkv = self.norm1(features_pos)  # Apply LayerNorm before attention
        attn_out, _ = self.attn(qkv, qkv, qkv)  # Self-attention
        features = features + attn_out  # Add residual connection

        # --- FFN with Pre-LayerNorm and Residual Connection ---
        ffn_in = self.norm2(features)  # Apply LayerNorm before FFN
        ffn_out = self.ffn(ffn_in)  # Apply FFN
        features = features + ffn_out  # Add residual connection

        return features


class CrossAttentionLayer(nn.Module):
    """Standard Transformer cross-attention block with pre-layer normalization."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        # Multi-Head Cross-Attention layer
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # Layer Normalization (applied before attention)
        self.norm_q = nn.LayerNorm(dim)  # Normalize query features
        self.norm_kv = nn.LayerNorm(dim)  # Normalize key/value features
        # self.norm_out = nn.LayerNorm(dim) # Often not needed/harmful with pre-norm? Check common practice. Using one after residual.
        self.norm_after_res = nn.LayerNorm(dim)  # Norm after residual

    def forward(self, feat_query, feat_kv):
        """
        Applies cross-attention where feat_query attends to feat_kv.

        Args:
            feat_query: (B, N_q, dim) - Query features (e.g., from left image).
            feat_kv: (B, N_kv, dim) - Key/Value features (e.g., from right image).

        Returns:
            features: (B, N_q, dim) - Output features after cross-attention.
            attn_weights: (B, N_q, N_kv) - Attention weights (optional).
        """
        # Apply LayerNorm before attention
        q = self.norm_q(feat_query)
        kv = self.norm_kv(feat_kv)

        # Cross-attention: query attends to key/value
        try:
            attn_out, attn_weights = self.attn(q, kv, kv)
        except Exception as e:
            print(f"[ERROR] Cross-Attention failed: {e}. Q shape: {q.shape}, KV shape: {kv.shape}")
            # Return query features as fallback
            return feat_query, torch.zeros((q.shape[0], q.shape[1], kv.shape[1]), device=q.device)

        # Add residual connection and apply output LayerNorm
        features = feat_query + attn_out
        features = self.norm_after_res(features)  # Apply norm after residual

        return features, attn_weights


# --- v2.8 ARCHITECTURE UPGRADE ---
class MatchingLayer(nn.Module):
    """
    一个交错的匹配层 (v2.8)
    包含一次共享的自注意力和一次双向的交叉注意力。
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        # 自注意力层 (用于更新自身上下文)
        # SOTA实践：左右图共享同一个自注意力层
        self.self_attn = SelfAttentionLayer(dim, num_heads)
        # 交叉注意力层 (用于与另一幅图像交换信息)
        self.cross_attn = CrossAttentionLayer(dim, num_heads)

    def forward(self, feat_left, pos_left, feat_right, pos_right):
        """
        feat_left: (B, N_l, C)
        feat_right: (B, N_r, C)
        pos_left: (B, N_l, C)
        pos_right: (B, N_r, C)
        """

        # 1. 自注意力 (左图更新自己, 右图更新自己)
        #    使用共享的 SelfAttentionLayer
        feat_left = self.self_attn(feat_left, pos_left)
        feat_right = self.self_attn(feat_right, pos_right)

        # 2. 交叉注意力 (双方交换信息)
        #    注意：交叉注意力层内部有 LayerNorm，所以我们传入更新前的 feat_left/right

        # 左图查询右图 (feat_left attend to feat_right)
        feat_left_updated, _ = self.cross_attn(feat_left, feat_right)

        # 右图查询左图 (feat_right attend to feat_left)
        feat_right_updated, _ = self.cross_attn(feat_right, feat_left)

        # 返回双向更新后的特征
        return feat_left_updated, feat_right_updated


class SparseMatchingNetwork(nn.Module):
    """
    (v2.8 - Interleaved Architecture)
    Performs sparse feature matching using INTERLEAVED self and cross attention layers.
    Predicts disparity based on feature similarity under epipolar constraints.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim = cfg.FEATURE_DIM
        num_layers = cfg.NUM_ATTENTION_LAYERS
        num_heads = cfg.NUM_HEADS
        print(f"Initializing Interleaved SparseMatchingNetwork with {num_layers} matching layers. (v2.8)")

        # Positional encoding module (不变)
        self.pos_enc = PositionalEncoding(dim)

        # --- v2.8 ARCHITECTURE CHANGE ---
        # 不再有 self_attn_left 和 self_attn_right
        # 而是有一个单一的、交错的匹配层列表
        self.matching_layers = nn.ModuleList([
            MatchingLayer(dim, num_heads) for _ in range(num_layers)
        ])

        # 移除了原有的 self.cross_attn，因为它现在包含在循环中
        # --- END v2.8 CHANGE ---

    def forward(self, desc_left, desc_right, kp_left, kp_right, image_size):
        """
        Matches keypoints between left and right images and predicts disparity.

        Args:
            desc_left: (B, N_l, C) - Left DINO descriptors.
            desc_right: (B, N_r, C) - Right DINO descriptors.
            kp_left: (B, N_l, 2) - Left keypoint coordinates (x, y).
            kp_right: (B, N_r, 2) - Right keypoint coordinates (x, y).
            image_size: (H, W) - Image dimensions.

        Returns:
            match_scores_constrained: (B, N_l, N_r) - Constrained matching scores (pre-softmax).
            disparity_pred: (B, N_l) - Predicted disparity for each left keypoint.
            constraint_mask: (B, N_l, N_r) - Boolean mask indicating geometrically valid matches.
        """
        # 1. Generate positional encodings (不变)
        pos_left = self.pos_enc(kp_left, image_size)
        pos_right = self.pos_enc(kp_right, image_size)

        # 初始特征 (不变)
        feat_left = desc_left
        feat_right = desc_right

        # --- v2.8 CORE LOOP CHANGE ---
        # 循环通过 N 个交错的匹配层
        # 在每一层，特征都会通过 Self-Attention 和 Cross-Attention 进行更新
        for layer in self.matching_layers:
            feat_left, feat_right = layer(feat_left, pos_left, feat_right, pos_right)

        # (移除了原有的 Self-Attention 循环和最终的 Cross-Attention)
        # --- END v2.8 CHANGE ---

        # 3. Compute raw matching scores (cosine similarity)
        #    此时的 feat_left 和 feat_right 已经充分融合了上下文和匹配信息
        eps = 1e-8  # Epsilon for numerical stability during normalization
        feat_left_norm = F.normalize(feat_left, dim=2, eps=eps)
        feat_right_norm = F.normalize(feat_right, dim=2, eps=eps)

        # (B, N_l, C) x (B, C, N_r) -> (B, N_l, N_r)
        match_scores = torch.bmm(feat_left_norm, feat_right_norm.transpose(1, 2))

        # 4. Enforce Epipolar Constraints (完全不变)
        # Get coordinate components for broadcasting
        x_left = kp_left[:, :, 0].unsqueeze(2)  # (B, N_l, 1)
        y_left = kp_left[:, :, 1].unsqueeze(2)  # (B, N_l, 1)
        x_right = kp_right[:, :, 0].unsqueeze(1)  # (B, 1, N_r)
        y_right = kp_right[:, :, 1].unsqueeze(1)  # (B, 1, N_r)

        # Constraint 1: Disparity must be non-negative (x_left >= x_right)
        valid_x_mask = (x_left >= x_right)
        # Constraint 2: Y-coordinates must be aligned within a threshold
        valid_y_mask = (y_left - y_right).abs() < self.cfg.DISPARITY_CONSTRAINT_Y_THRESHOLD
        # Combine constraints: a match is valid only if both X and Y constraints are met
        constraint_mask = (valid_x_mask & valid_y_mask)  # (B, N_l, N_r)

        # Apply the mask to the scores: invalid matches get a very low score
        match_scores_constrained = match_scores.masked_fill(~constraint_mask, torch.finfo(match_scores.dtype).min)

        # 5. Predict Disparity using Soft Argmax (Weighted Average) (完全不变)
        # Apply temperature scaling and softmax to get probabilities
        match_probs = F.softmax(match_scores_constrained * self.cfg.MATCHING_TEMPERATURE, dim=2)  # (B, N_l, N_r)
        # v2.2 NaN Fix
        match_probs = torch.nan_to_num(match_probs, nan=0.0)

        # Calculate the disparity matrix: disparity(i, j) = x_left(i) - x_right(j)
        # (B, N_l, 1) - (B, 1, N_r) -> (B, N_l, N_r)
        disparity_matrix = kp_left[:, :, 0].unsqueeze(2) - kp_right[:, :, 0].unsqueeze(1)

        # Compute the expected disparity for each left keypoint as a weighted average
        # Weighted sum: Sum_j [ probability(match i -> j) * disparity(i, j) ]
        disparity_pred = (disparity_matrix * match_probs).sum(dim=2)  # (B, N_l)

        # Handle potential NaNs if a left keypoint had NO valid matches after constraints
        disparity_pred = torch.nan_to_num(disparity_pred, nan=0.0)  # Replace NaN with 0 disparity

        # Return constrained scores (needed by loss), predicted disparity, and the mask
        return match_scores_constrained, disparity_pred, constraint_mask


# --- 4. NEW: Feature-Metric Loss ---
class FeatureMetricLoss(nn.Module):
    """
    Computes loss based on DINOv3 feature similarity (Cosine Similarity),
    making it robust to non-Lambertian surfaces and sparse data, unlike pixel-based losses.
    Includes a smoothness term to regularize disparity predictions.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, desc_left, desc_right, match_scores, keypoints_left, disparity, scores_left, constraint_mask):
        """
        Calculates the combined feature similarity loss and smoothness loss.

        Args:
            desc_left: (B, N_l, C) Left DINO descriptors.
            desc_right: (B, N_r, C) Right DINO descriptors.
            match_scores: (B, N_l, N_r) Constrained matching scores (pre-softmax) from the matcher.
            keypoints_left: (B, N_l, 2) Left keypoint coordinates.
            disparity: (B, N_l) Predicted disparity for left keypoints.
            scores_left: (B, N_l) Scores indicating the validity/confidence of left keypoints.
            constraint_mask: (B, N_l, N_r) Boolean mask of geometrically valid matches.

        Returns:
            feature_loss: Scalar tensor representing the average feature similarity loss.
            smooth_loss: Scalar tensor representing the average unweighted smoothness loss.
        """
        B, N_l, C = desc_left.shape
        device = desc_left.device

        # --- 1. Feature Similarity Loss (Cosine Similarity Loss) ---

        # Apply temperature and softmax to constrained scores to get matching probabilities
        # This mirrors the calculation in the matcher network.
        match_probs = F.softmax(match_scores * self.cfg.MATCHING_TEMPERATURE, dim=2)  # (B, N_l, N_r)

        # Fix potential NaN issue: If a row in match_scores was all -inf (no valid matches),
        # softmax results in NaN. Replace these NaNs with 0 probability.
        match_probs = torch.nan_to_num(match_probs, nan=0.0)

        # Calculate the expected/"soft-matched" right descriptor for each left descriptor
        # Weighted average of right descriptors based on matching probabilities.
        # (B, N_l, N_r) @ (B, N_r, C) -> (B, N_l, C)
        desc_right_weighted = torch.bmm(match_probs, desc_right)

        # Normalize features for cosine similarity calculation
        # Add epsilon for numerical stability, preventing division by zero for zero vectors.
        eps = 1e-8
        desc_left_norm = F.normalize(desc_left, dim=2, eps=eps)
        desc_right_weighted_norm = F.normalize(desc_right_weighted, dim=2, eps=eps)

        # Calculate cosine similarity between left descriptor and its expected right counterpart
        # Higher similarity is better. Dot product of normalized vectors.
        cosine_sim = (desc_left_norm * desc_right_weighted_norm).sum(dim=2)  # (B, N_l)
        # Convert similarity to a loss: minimize (1 - similarity)
        # Loss is low when similarity is high (near 1), high when low (near -1 or 0).
        feature_loss_per_kp = 1.0 - cosine_sim  # (B, N_l), ranges roughly [0, 2]

        # --- Masking for valid keypoints ---
        # Create a mask for keypoints that should contribute to the loss:
        # 1. Must be a valid keypoint detected (score > 0.0, as per v2.13 detector fix)
        # detection_mask = (scores_left > 0.0)  # (B, N_l)

        # =========================================================================
        # --- [START] MODIFICATION: 修复 Baseline 的检测器阈值 ---
        # =========================================================================
        # 斑点检测器返回的是 'size' (最小为 10), 而不是亮度
        detection_mask = (scores_left > 0.1)  # (B, N_l)
        # =========================================================================
        # --- [END] MODIFICATION ---
        # =========================================================================

        # 2. Must have had at least one geometrically valid match candidate in the right image
        matchable_mask = torch.any(constraint_mask, dim=2)  # (B, N_l)

        # Combine masks: only compute loss for detected and matchable keypoints
        final_valid_mask = detection_mask & matchable_mask  # (B, N_l)

        # Apply the mask to the loss per keypoint
        masked_loss = feature_loss_per_kp * final_valid_mask

        # Calculate the average loss over valid keypoints
        num_valid = torch.sum(final_valid_mask)
        feature_loss = torch.tensor(0.0, device=device)  # Default to 0 if no valid points
        if num_valid > 0:
            feature_loss = masked_loss.sum() / num_valid

        # --- 2. Smoothness Loss (Disparity Regularization) ---
        # Calculate the unweighted smoothness loss (average disparity difference between neighbors)
        smooth_loss = self._compute_sparse_smoothness(keypoints_left, disparity, scores_left)

        # Return the unweighted feature loss and unweighted smoothness loss
        # The weighting happens in the Trainer class.
        return feature_loss, smooth_loss

    def _compute_sparse_smoothness(self, keypoints, disparity, scores):
        """
        Calculates the smoothness loss: encourages nearby valid keypoints
        to have similar disparity values. Returns the unweighted average loss.

        Args:
            keypoints: (B, N, 2) Keypoint coordinates.
            disparity: (B, N) Predicted disparity.
            scores: (B, N) Keypoint detection scores.

        Returns:
            Scalar tensor representing the average smoothness loss (unweighted).
        """
        B, N, _ = keypoints.shape
        total_smooth_loss = 0.0
        total_valid_pairs = 0

        for b in range(B):  # Process each item in the batch
            kp = keypoints[b]  # (N, 2)
            disp = disparity[b]  # (N,)
            sc = scores[b]  # (N,)

            # Filter only valid (detected) keypoints
            valid_mask = sc > 0.0  # (Use 0.0, as per v2.13)
            # --- MODIFICATION: 斑点检测器的分数是 size, 最小是 10 ---
            valid_mask = sc > 0.1  # (Baseline 使用 0.1, 匹配斑点大小)
            # --- END MODIFICATION ---
            kp_valid = kp[valid_mask]  # (M, 2)
            disp_valid = disp[valid_mask]  # (M,)

            # Need at least 2 points to compute pairwise differences
            if len(kp_valid) < 2:
                continue

            # Compute pairwise Euclidean distances between all valid keypoints
            # (M, 1, 2) - (1, M, 2) -> (M, M, 2) -> norm -> (M, M)
            dist = torch.cdist(kp_valid, kp_valid)

            # Identify neighboring pairs (distance < threshold, excluding self-pairs)
            neighbor_mask = (dist < 20) & (dist > 1e-6)  # Threshold = 20 pixels

            # Calculate absolute difference in disparity between all pairs
            # (M, 1) - (1, M) -> (M, M)
            disp_diff = (disp_valid.unsqueeze(1) - disp_valid.unsqueeze(0)).abs()

            # Sum the disparity differences only for neighboring pairs
            batch_smooth_loss = (disp_diff * neighbor_mask.float()).sum()
            num_valid_pairs = neighbor_mask.sum()

            total_smooth_loss += batch_smooth_loss
            total_valid_pairs += num_valid_pairs

        # Calculate the average smoothness loss across the batch
        smooth_loss_avg = torch.tensor(0.0, device=keypoints.device)
        if total_valid_pairs > 0:
            smooth_loss_avg = total_smooth_loss / total_valid_pairs

        # Return the raw, unweighted smoothness loss average
        return smooth_loss_avg


# --- 5. Dataset ---
class RectifiedWaveStereoDataset(Dataset):
    """ Loads, rectifies, crops, resizes, and augments stereo image pairs. """

    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.is_validation = is_validation  # Flag to disable augmentation for validation set

        # Find all left images
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"错误: 在 '{cfg.LEFT_IMAGE_DIR}' 中未找到图像。")

        # Load stereo calibration parameters
        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            # Rectification maps
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            # Regions of Interest (ROI) after rectification
            self.roi_left = tuple(map(int, calib['roi_left']))  # Ensure integer coordinates
            self.roi_right = tuple(map(int, calib['roi_right']))

            # =========================================================================
            # --- [START] MODIFICATION: 恢复 Baseline 的自动分辨率 (裁剪) 逻辑 ---
            # =========================================================================
            # (从 ...1030_gemini.py 移植而来)
            _x_l, _y_l, w_l, h_l = self.roi_left
            _x_r, _y_r, w_r, h_r = self.roi_right

            if self.cfg.IMAGE_WIDTH == 0 or self.cfg.IMAGE_HEIGHT == 0:
                print(f"[Dataset] Left ROI: {w_l}x{h_l}, Right ROI: {w_r}x{h_r}")
                target_w = min(w_l, w_r)
                target_h = min(h_l, h_r)

                # 确保分辨率是 16 的倍数 (DINO patch size)
                patch_size = 16
                target_w = (target_w // patch_size) * patch_size
                target_h = (target_h // patch_size) * patch_size

                print(f"[Dataset] 自动设置目标分辨率 (最小公共 ROI, 16x 对齐) 为: {target_w}x{target_h}")
                self.cfg.IMAGE_WIDTH = target_w
                self.cfg.IMAGE_HEIGHT = target_h
            # =========================================================================
            # --- [END] MODIFICATION ---
            # =========================================================================

        except Exception as e:
            sys.exit(f"加载标定文件 '{cfg.CALIBRATION_FILE}' 失败: {e}")

        # Split dataset into training and validation sets
        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        # Use a fixed random seed for reproducibility of splits
        np.random.seed(42)
        np.random.shuffle(indices)  # Shuffle indices for random split
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]
        print(f"{'验证集' if is_validation else '训练集'}: {len(self.indices)} 帧")

    def __len__(self):
        """ Returns the number of samples in the dataset (train or val). """
        return len(self.indices)

    def __getitem__(self, idx):
        """ Loads, preprocesses, and returns a single stereo pair. """
        try:
            # Get image paths based on the shuffled index
            img_idx = self.indices[idx]
            if img_idx < 0 or img_idx >= len(self.left_images):
                print(f"警告: 无效的文件索引 {img_idx} (来自 dataset index {idx})。")
                return None

            left_path = self.left_images[img_idx]
            # Construct right image path based on left image name convention
            right_filename = 'right' + os.path.basename(left_path)[4:]
            right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_filename)

            # Load images in grayscale
            left_raw = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            right_raw = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

            if left_raw is None or right_raw is None:
                print(f"警告: 无法加载图像对 idx {idx} (file index {img_idx}): {left_path}, {right_path}")
                return None  # Skip this sample if loading fails

            # --- Image Preprocessing ---
            # 1. Rectification: Apply precomputed undistortion and rectification maps
            left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

            # 2. ROI Cropping: Extract the valid region after rectification
            lx, ly, lw, lh = self.roi_left
            rx, ry, rw, rh = self.roi_right
            # Add boundary checks for ROI
            if ly + lh > left_rect.shape[0] or lx + lw > left_rect.shape[1] or \
                    ry + rh > right_rect.shape[0] or rx + rw > right_rect.shape[1] or \
                    ly < 0 or lx < 0 or ry < 0 or rx < 0:
                print(f"警告: ROI 坐标无效或超出图像边界 (idx {idx}, file index {img_idx})。跳过。")
                print(f"  Left ROI: {self.roi_left}, Left Shape: {left_rect.shape}")
                print(f"  Right ROI: {self.roi_right}, Right Shape: {right_rect.shape}")
                return None

            left_rect_cropped = left_rect[ly:ly + lh, lx:lx + lw]
            right_rect_cropped = right_rect[ry:ry + rh, rx:rx + rw]

            # 3. Resizing: Resize cropped images to the network's input size
            # target_size = (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT)

            # =========================================================================
            # --- [START] MODIFICATION: 恢复 Baseline 的 "裁剪" 逻辑 ---
            # =========================================================================
            # (从 ...1030_gemini.py 移植而来)
            # 我们进行裁剪 (Crop), 而不是缩放 (Resize), 以保留像素
            if left_rect_cropped.shape[0] < self.cfg.IMAGE_HEIGHT or left_rect_cropped.shape[1] < self.cfg.IMAGE_WIDTH:
                print(
                    f"警告: ROI {left_rect_cropped.shape} 小于目标裁剪尺寸 ({self.cfg.IMAGE_HEIGHT}, {self.cfg.IMAGE_WIDTH})。跳过。")
                return None
            if right_rect_cropped.shape[0] < self.cfg.IMAGE_HEIGHT or right_rect_cropped.shape[
                1] < self.cfg.IMAGE_WIDTH:
                print(
                    f"警告: ROI {right_rect_cropped.shape} 小于目标裁剪尺寸 ({self.cfg.IMAGE_HEIGHT}, {self.cfg.IMAGE_WIDTH})。跳过。")
                return None

            left_img = left_rect_cropped[0:self.cfg.IMAGE_HEIGHT, 0:self.cfg.IMAGE_WIDTH]
            right_img = right_rect_cropped[0:self.cfg.IMAGE_HEIGHT, 0:self.cfg.IMAGE_WIDTH]
            # =========================================================================
            # --- [END] MODIFICATION ---
            # =========================================================================

            # if left_rect_cropped.shape[0] == 0 or left_rect_cropped.shape[1] == 0 or \
            #         right_rect_cropped.shape[0] == 0 or right_rect_cropped.shape[1] == 0:
            #     print(f"警告: ROI 裁剪后图像为空 (idx {idx}, file index {img_idx})。跳过。")
            #     return None
            #
            # left_img = cv2.resize(left_rect_cropped, target_size)
            # right_img = cv2.resize(right_rect_cropped, target_size)

            # 4. Augmentation (only during training)
            if not self.is_validation and self.cfg.USE_ADVANCED_AUGMENTATION and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
                # Random brightness adjustment
                if np.random.rand() < 0.5:
                    brightness = np.random.uniform(0.7, 1.3)
                    left_img = np.clip(left_img * brightness, 0, 255).astype(np.uint8)
                    right_img = np.clip(right_img * brightness, 0, 255).astype(np.uint8)
                # Random contrast adjustment
                if np.random.rand() < 0.5:
                    contrast = np.random.uniform(0.7, 1.3)
                    mean_l, mean_r = left_img.mean(), right_img.mean()
                    left_img = np.clip((left_img - mean_l) * contrast + mean_l, 0, 255).astype(np.uint8)
                    right_img = np.clip((right_img - mean_r) * contrast + mean_r, 0, 255).astype(np.uint8)

            # --- v2.12 MASK FIX ---
            # 5. Mask Generation:
            # 移除了有害的 cv2.threshold。
            # SparseKeypointDetector 应该在完整的图像上运行，
            # 并自己找出最亮的 top-k 点。
            # 我们创建一个“全通”掩码。
            # mask = np.ones_like(left_img, dtype=np.uint8) * 255
            # --- END v2.12 FIX ---

            # =========================================================================
            # --- [START] MODIFICATION: 恢复 Baseline 的掩码逻辑 ---
            # =========================================================================
            # SimpleBlobDetector 需要一个二值掩码 (来自 baseline ...1030_gemini.py)
            # MASK_THRESHOLD 在 Config 中定义为 30
            # _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
            # =========================================================================
            # --- [END] MODIFICATION: 恢复 Baseline 的掩码逻辑 ---
            # =========================================================================

            # =========================================================================
            # --- [START] BUG FIX: 移除有害的 cv2.threshold 掩码 ---
            # =========================================================================
            # 错误: 上面的 cv2.threshold 掩码是多余的, 且会破坏 SimpleBlobDetector 的
            # 内部阈值逻辑。
            # 正确: 我们应该使用一个“全通”掩码, 让检测器在完整的灰度图上运行。
            mask = np.ones_like(left_img, dtype=np.uint8) * 255
            # =========================================================================
            # --- [END] BUG FIX ---
            # =========================================================================

            # --- v2.10: 移除了 v2.9 的膨胀步骤 ---
            left_img_for_tensors = left_img
            right_img_for_tensors = right_img
            mask_for_tensors = mask
            # --- END v2.10 MODIFICATION ---

            # 6. --- Convert to Tensors ---
            # Normalize grayscale images to [0, 1]
            left_gray = torch.from_numpy(left_img_for_tensors).float().unsqueeze(0) / 255.0  # (1, H, W)
            right_gray = torch.from_numpy(right_img_for_tensors).float().unsqueeze(0) / 255.0

            # Convert grayscale to RGB for DINOv3 input, normalize to [0, 1]
            # Output shape: (3, H, W)
            left_rgb = torch.from_numpy(
                cv2.cvtColor(left_img_for_tensors, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0
            right_rgb = torch.from_numpy(
                cv2.cvtColor(right_img_for_tensors, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0

            # Convert mask to tensor, normalize to [0, 1]
            mask_tensor = torch.from_numpy(mask_for_tensors).float().unsqueeze(0) / 255.0  # (1, H, W)

            # Return dictionary of tensors
            return {
                'left_gray': left_gray,  # For Keypoint Detector
                'right_gray': right_gray,  # For Keypoint Detector (right)
                'left_rgb': left_rgb,  # For DINO Feature Extractor
                'right_rgb': right_rgb,  # For DINO Feature Extractor (right)
                'mask': mask_tensor  # For Keypoint Detector (left mask)
            }
        except Exception as e:
            # Catch potential errors during loading/processing and return None
            # The collate_fn will handle filtering these out.
            print(f"警告: 处理索引 {idx} (文件索引 {img_idx}) 时发生意外错误: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return None


# --- 6. Complete Model ---
class SparseMatchingStereoModel(nn.Module):
    """ The complete sparse matching stereo model pipeline. """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Initialize sub-modules
        self.keypoint_detector = SparseKeypointDetector(cfg)
        self.feature_extractor = DINOv3FeatureExtractor(cfg)
        self.matcher = SparseMatchingNetwork(cfg)  # This will now initialize the v2.8 matcher

        print(f"Sparse Matching Model: max {cfg.MAX_KEYPOINTS} keypoints, "
              f"{cfg.NUM_ATTENTION_LAYERS} attention layers")
        print(f"Using NEW Feature-Metric Loss (Cosine Similarity)")

    def forward(self, left_gray, right_gray, left_rgb, right_rgb, mask):
        """
        Processes a stereo pair through the entire pipeline.

        Args:
            left_gray: (B, 1, H, W) - Left grayscale image for keypoint detection.
            right_gray: (B, 1, H, W) - Right grayscale image for keypoint detection.
            left_rgb: (B, 3, H, W) - Left RGB image for DINO feature extraction.
            right_rgb: (B, 3, H, W) - Right RGB image for DINO feature extraction.
            mask: (B, 1, H, W) - Mask for left keypoint detection.

        Returns:
            Dictionary containing intermediate and final outputs:
            - keypoints_left, keypoints_right: (B, N, 2) Coordinates.
            - scores_left, scores_right: (B, N) Detection scores.
            - descriptors_left, descriptors_right: (B, N, C) DINO features.
            - match_scores: (B, N_l, N_r) Constrained scores from matcher.
            - disparity: (B, N_l) Predicted disparity.
            - constraint_mask: (B, N_l, N_r) Boolean mask of valid matches.
        """
        B, _, H, W = left_gray.shape

        # 1. Detect sparse keypoints in both images
        #    (v2.12: 在全通掩码上运行)
        kp_left, scores_left = self.keypoint_detector(left_gray, mask)  # Use mask for left
        # Detect on the full right gray image (no explicit right mask used)
        # kp_right, scores_right = self.keypoint_detector(right_gray, torch.ones_like(right_gray))

        # =========================================================================
        # --- [START] MODIFICATION: 修复 Baseline 的右图检测器 bug ---
        # =========================================================================
        # (从 ...1030_gemini.py 移植而来)
        # 右图检测器也应该使用 (来自左图的) 掩码
        kp_right, scores_right = self.keypoint_detector(right_gray, mask)
        # =========================================================================
        # --- [END] MODIFICATION ---
        # =========================================================================

        # 2. Extract DINO features at keypoint locations
        desc_left = self.feature_extractor(left_rgb, kp_left)
        desc_right = self.feature_extractor(right_rgb, kp_right)

        # 3. Match features using the attention-based matcher network (v2.8)
        # Returns constrained scores, predicted disparity, and the constraint mask
        match_scores, disparity, constraint_mask = self.matcher(
            desc_left, desc_right, kp_left, kp_right, (H, W)
        )

        # Return all relevant outputs in a dictionary
        return {
            'keypoints_left': kp_left,
            'keypoints_right': kp_right,
            'scores_left': scores_left,
            'scores_right': scores_right,
            'descriptors_left': desc_left,
            'descriptors_right': desc_right,
            'match_scores': match_scores,  # Constrained scores
            'disparity': disparity,  # Predicted disparity
            'constraint_mask': constraint_mask  # Mask used for scores and loss
        }


# --- 7. Evaluation Metrics ---
class EvaluationMetrics:
    """ Computes evaluation metrics based on sparse disparity predictions. """

    @staticmethod
    def compute_sparse_metrics(disparity, keypoints_left, scores_left):
        """
        Computes basic metrics like mean/std disparity and number of valid keypoints.

        Args:
            disparity: (B, N) - Predicted disparity tensor.
            keypoints_left: (B, N, 2) - Left keypoint coordinates.
            scores_left: (B, N) - Left keypoint detection scores.

        Returns:
            Dictionary containing computed metrics.
        """
        metrics = {}
        B, N = disparity.shape

        # Create mask for valid (detected) keypoints
        valid_mask = scores_left > 0.0  # Shape (B, N) (Use 0.0, as per v2.13)
        # --- MODIFICATION: 斑点检测器的分数是 size, 最小是 10 ---
        valid_mask = scores_left > 0.1  # (Baseline 使用 0.1, 匹配斑点大小)
        # --- END MODIFICATION ---

        # Flatten batch and mask for easier computation
        disparity_flat = disparity[valid_mask]  # 1D tensor of valid disparities

        if disparity_flat.numel() > 0:
            # Filter out potential NaNs or Infs from disparity prediction
            finite_disp = disparity_flat[torch.isfinite(disparity_flat)]
            if finite_disp.numel() > 0:
                # Compute mean and standard deviation on finite values
                metrics['mean_disparity'] = finite_disp.mean().item()
                metrics['std_disparity'] = finite_disp.std().item()
            else:
                # Handle case where all valid disparities are non-finite
                metrics['mean_disparity'] = 0.0
                metrics['std_disparity'] = 0.0
            # Calculate total number of valid keypoints across the batch
            metrics['num_valid_keypoints'] = valid_mask.sum().item() / B  # Average per image
        else:
            # Handle case with no valid keypoints
            metrics['mean_disparity'] = 0.0
            metrics['std_disparity'] = 0.0
            metrics['num_valid_keypoints'] = 0

        return metrics


# --- Collate Function ---
def collate_fn(batch):
    """ Custom collate function to filter out None samples from the batch. """
    # Remove samples where __getitem__ returned None (e.g., due to loading errors)
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None  # Return None if the entire batch is invalid
    # Use the default collate function on the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)


# --- 8. Trainer ---
class Trainer:
    """ Handles the training and validation loops, logging, visualization, and checkpointing. """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Generate timestamp for the current run
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        # Create directory structure for the run
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR,
                                    self.timestamp + f"_v2.13_dense_kpts_{cfg.MAX_KEYPOINTS}")  # MODIFIED run name
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.vis_dir = os.path.join(self.run_dir, "visualizations")
        self.log_dir_json = os.path.join(self.run_dir, "logs")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")
        for d in [self.ckpt_dir, self.vis_dir, self.log_dir_json, self.tb_dir]:
            os.makedirs(d, exist_ok=True)

        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize datasets and dataloaders
        train_ds = RectifiedWaveStereoDataset(cfg, is_validation=False)
        val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)

        # --- MODIFICATION: 解决 Windows 上的 num_workers=0 瓶颈 ---
        # 强制 num_workers = 4 来异步加载数据，防止 GPU 空闲
        num_workers = 4
        # --- 移除这里的 print ---
        # print(f"[Trainer Config] 强制设置 num_workers={num_workers} (原 Windows 默认为 0)")
        # --- END MODIFICATION ---

        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_fn, num_workers=num_workers,
                                       pin_memory=True if self.device.type == 'cuda' else False,
                                       persistent_workers=True if num_workers > 0 else False)  # 添加 persistent_workers

        # =========================================================================
        # --- [START] MODIFICATION: 修复 Val Loader 卡住的 Bug ---
        # =========================================================================
        # persistent_workers=True 配合 shuffle=False 在 Windows 上会导致
        # 验证集迭代器无法重置，永远返回相同的数据。
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=num_workers,
                                     pin_memory=True if self.device.type == 'cuda' else False,
                                     persistent_workers=False)  # <--- 必须设为 False
        # =========================================================================
        # --- [END] MODIFICATION ---
        # =========================================================================

        # Initialize TensorBoard writer if available
        try:
            self.writer = SummaryWriter(log_dir=self.tb_dir) if SummaryWriter else None
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard SummaryWriter: {e}")
            self.writer = None

        # Initialize the model and move it to the device
        self.model = SparseMatchingStereoModel(cfg).to(self.device)
        # Initialize the loss function
        self.loss_fn = FeatureMetricLoss(cfg)

        # Initialize optimizer (AdamW) - only optimize parameters with requires_grad=True
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=cfg.LEARNING_RATE, weight_decay=1e-4)  # Added weight decay

        # Initialize learning rate scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-7  # Anneal LR down to 1e-7
        )

        # Initialize evaluation metrics handler
        self.evaluator = EvaluationMetrics()
        # Initialize gradient scaler for mixed-precision training
        self.scaler = torch.amp.GradScaler(enabled=cfg.USE_MIXED_PRECISION and self.device.type == 'cuda')
        self.step = 0  # Global step counter
        self.log_file = os.path.join(self.log_dir_json, "training_log.json")  # Path for JSON log

        # Define keys for logging losses and metrics
        self.loss_keys = ['total', 'feature', 'smoothness']  # Updated for FeatureMetricLoss
        self.metric_keys = ['mean_disparity', 'std_disparity', 'num_valid_keypoints']
        # Dictionary to store training history
        self.history = {
            'train': {k: [] for k in self.loss_keys + self.metric_keys},
            'val': {k: [] for k in self.loss_keys + self.metric_keys}
        }
        # Save config to log file at the beginning
        self.update_log_file(-1)  # Log initial config before epoch 0

    def train(self):
        """ Main training loop over epochs. """
        print(f"\n--- Starting Sparse Matching Training (Original Sparse Config) ---")  # MODIFIED
        print(
            f"  Resolution: {self.cfg.IMAGE_WIDTH}x{self.cfg.IMAGE_HEIGHT}, Grad Accum: {self.cfg.GRADIENT_ACCUMULATION_STEPS}")
        print(
            f"  Config: Max keypoints: {self.cfg.MAX_KEYPOINTS}, NMS: {self.cfg.NMS_RADIUS}, Layers: {self.cfg.NUM_ATTENTION_LAYERS}, LR: {self.cfg.LEARNING_RATE:.1e}, Smooth Weight: {self.cfg.SMOOTHNESS_WEIGHT:.1e}")
        print(f"  [Info] 恢复为原始稀疏设置 (MaxKpts: {self.cfg.MAX_KEYPOINTS})。")  # MODIFIED
        best_val_loss = float('inf')  # Track best validation loss
        epochs_no_improve = 0  # Counter for early stopping

        # --- Graceful Exit Handling ---
        try:
            for epoch in range(self.cfg.NUM_EPOCHS):
                # --- Training Epoch ---
                train_results = self._run_epoch(epoch, is_training=True)
                # Check for empty results (can happen if loader fails completely)
                if not train_results:
                    print(f"Epoch {epoch + 1}: Training epoch failed or returned empty results. Stopping.")
                    break
                self._log_epoch_results('train', epoch, train_results)  # Log results

                # --- Validation Epoch ---
                with torch.no_grad():  # Disable gradients during validation
                    val_results = self._run_epoch(epoch, is_training=False)
                if not val_results:
                    print(f"Epoch {epoch + 1}: Validation epoch failed or returned empty results. Stopping.")
                    break
                self._log_epoch_results('val', epoch, val_results)  # Log results

                # Get current validation loss
                current_val_loss = val_results.get('total', float('inf'))

                # Print epoch summary
                print(
                    f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> "
                    f"Train Loss: {train_results.get('total', float('nan')):.4f} (feat: {train_results.get('feature', float('nan')):.4f}) | "
                    f"Val Loss: {current_val_loss:.4f} (feat: {val_results.get('feature', float('nan')):.4f}) | "
                    f"Val Keypoints: {val_results.get('num_valid_keypoints', 0):.0f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

                # Update JSON log file
                self.update_log_file(epoch)
                # Plot training history if enabled
                if self.cfg.VISUALIZE_TRAINING:
                    try:
                        self.plot_training_history()
                    except Exception as plot_e:
                        print(f"Warning: Failed to plot training history for epoch {epoch + 1}: {plot_e}")

                # --- Checkpoint Saving and Early Stopping ---
                if current_val_loss < best_val_loss:
                    # Improvement found
                    best_val_loss = current_val_loss
                    epochs_no_improve = 0  # Reset counter
                    # Save the best model state dictionary
                    save_path = os.path.join(self.ckpt_dir, "best_model_sparse.pth")
                    try:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"  Val loss improved to {best_val_loss:.4f}. Model saved to {save_path}")
                    except Exception as save_e:
                        print(f"Warning: Failed to save model checkpoint: {save_e}")

                else:
                    # No improvement
                    epochs_no_improve += 1
                    print(f"  No validation loss improvement for {epochs_no_improve} epochs.")
                    if epochs_no_improve >= self.cfg.EARLY_STOPPING_PATIENCE:
                        # Trigger early stopping
                        print(
                            f"--- Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss for {self.cfg.EARLY_STOPPING_PATIENCE} epochs. ---")
                        break  # Exit training loop

                # Step the learning rate scheduler
                self.scheduler.step()

        except KeyboardInterrupt:
            print("\n--- Training interrupted by user (KeyboardInterrupt). ---")
        except Exception as train_e:
            print(f"\n--- Training failed due to an error: {train_e} ---")
            import traceback
            traceback.print_exc()  # Print full traceback
        finally:
            print("--- Training loop finished or stopped. ---")
            # Close TensorBoard writer if used
            if self.writer:
                print("Closing TensorBoard writer...")
                self.writer.close()
            print("--- Training script exiting. ---")

    def _pad_inputs(self, *tensors):
        """ Pads input tensors height and width to be divisible by the patch size (16 for DINO). """
        if not tensors: return []  # Handle empty input
        # Get dimensions from the first tensor
        b, c, h, w = tensors[0].shape
        patch_size = 16  # DINOv3 patch size
        # Calculate padding needed for height and width
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        # If padding is needed, apply it to all tensors
        if pad_h > 0 or pad_w > 0:
            # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
            padded_tensors = [F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0) for x in tensors]
            return padded_tensors
        # Return original tensors if no padding is needed
        return list(tensors)

    def _run_epoch(self, epoch, is_training):
        """ Runs a single epoch of training or validation. """
        # Set model mode (train or eval)
        self.model.train(is_training)
        # Select appropriate dataloader
        loader = self.train_loader if is_training else self.val_loader
        # Initialize progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [{'Train' if is_training else 'Val'}]",
                    leave=True)  # Keep pbar after loop
        # Dictionary to accumulate results for the epoch
        epoch_results = {k: 0.0 for k in self.loss_keys + self.metric_keys}
        batch_count = 0

        # Loop over batches in the dataloader
        for data in pbar:
            # Skip if collate_fn returned None (empty batch)
            if data is None:
                print("Warning: Skipping None batch returned by collate_fn.")
                continue

            # Move data tensors to the designated device
            try:
                left_gray = data['left_gray'].to(self.device, non_blocking=True)
                right_gray = data['right_gray'].to(self.device, non_blocking=True)
                left_rgb = data['left_rgb'].to(self.device, non_blocking=True)
                right_rgb = data['right_rgb'].to(self.device, non_blocking=True)
                mask = data['mask'].to(self.device, non_blocking=True)
            except KeyError as e:
                print(f"Error: Missing key {e} in data dictionary. Skipping batch.")
                continue
            except Exception as e:
                print(f"Error moving data to device: {e}. Skipping batch.")
                continue

            # Pad inputs to be compatible with DINO patch size
            try:
                left_gray, right_gray, left_rgb, right_rgb, mask = self._pad_inputs(
                    left_gray, right_gray, left_rgb, right_rgb, mask
                )
            except Exception as e:
                print(f"Error padding inputs: {e}. Skipping batch.")
                continue

            # Automatic Mixed Precision (AMP) context for potential speedup
            autocast_kwargs = {'device_type': self.device.type, 'enabled': self.cfg.USE_MIXED_PRECISION}

            with torch.autocast(**autocast_kwargs):
                # --- Forward pass ---
                try:
                    outputs = self.model(left_gray, right_gray, left_rgb, right_rgb, mask)
                    if outputs is None:  # Added check
                        raise ValueError("Model forward pass returned None")
                except Exception as e:
                    print(f"\nError during model forward pass: {e}. Skipping batch.")
                    if is_training: self.optimizer.zero_grad()  # Clear any partial grads
                    continue  # Skip to next batch

                # --- Loss calculation ---
                try:
                    # Pass all necessary outputs to the FeatureMetricLoss
                    feature_loss, smooth_loss_unweighted = self.loss_fn(
                        outputs['descriptors_left'],
                        outputs['descriptors_right'],
                        outputs['match_scores'],
                        outputs['keypoints_left'],
                        outputs['disparity'],
                        outputs['scores_left'],
                        outputs['constraint_mask']  # Pass the constraint mask
                    )

                    # Calculate total weighted loss
                    total_loss = (self.cfg.PHOTOMETRIC_WEIGHT * feature_loss +
                                  self.cfg.SMOOTHNESS_WEIGHT * smooth_loss_unweighted)

                    if not torch.isfinite(total_loss):  # Check before backward
                        raise ValueError("Loss became NaN or Inf during calculation.")

                except KeyError as e:
                    print(f"\nError: Missing key {e} in model outputs during loss calculation. Skipping batch.")
                    if is_training: self.optimizer.zero_grad()
                    continue
                except Exception as e:
                    print(f"\nError calculating loss: {e}. Skipping batch.")
                    if is_training: self.optimizer.zero_grad()
                    continue

            # --- Backpropagation and Optimization (only during training) ---
            if is_training:
                accum_steps = self.cfg.GRADIENT_ACCUMULATION_STEPS
                # Scale the loss for AMP and gradient accumulation
                # Move scaling outside autocast context
                scaled_loss = self.scaler.scale(total_loss / accum_steps)

                try:
                    # Backpropagate the scaled loss
                    scaled_loss.backward()

                    # Perform optimizer step only after accumulating enough steps
                    if (self.step + 1) % accum_steps == 0:
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                        # Optimizer step (updates model parameters)
                        self.scaler.step(self.optimizer)
                        # Update the scaler for the next iteration
                        self.scaler.update()
                        # Zero gradients for the next accumulation cycle
                        self.optimizer.zero_grad()
                except Exception as e:
                    print(f"\nError during backpropagation or optimizer step: {e}. Skipping step.")
                    # Zero gradients to prevent accumulation of bad gradients
                    self.optimizer.zero_grad()

            # --- Compute and Accumulate Metrics ---
            try:
                metrics = self.evaluator.compute_sparse_metrics(
                    outputs['disparity'].detach(),  # Detach tensors for metric calculation
                    outputs['keypoints_left'].detach(),
                    outputs['scores_left'].detach()
                )
            except Exception as e:
                print(f"\nWarning: Failed to compute metrics: {e}. Metrics will be zero.")
                metrics = {k: 0.0 for k in self.metric_keys}

            # Accumulate results for epoch average (use .item() to get Python numbers)
            epoch_results['total'] += total_loss.item()  # Already checked for finite
            epoch_results['feature'] += feature_loss.item()
            epoch_results['smoothness'] += smooth_loss_unweighted.item()  # Store unweighted

            for k in self.metric_keys:
                epoch_results[k] += metrics.get(k, 0.0)  # Use .get for safety

            batch_count += 1  # Increment batch counter

            # Update progress bar description with current batch stats
            pbar.set_postfix({
                'loss': total_loss.item(),
                'feat_loss': feature_loss.item(),
                'kpts': metrics.get('num_valid_keypoints', 0),
                'disp': metrics.get('mean_disparity', 0.0)
            })

            # --- Logging and Visualization (only during training) ---
            if is_training:
                # Log step loss to TensorBoard
                if self.writer:
                    self.writer.add_scalar('Loss/step_train_total', total_loss.item(), self.step)
                    self.writer.add_scalar('Loss/step_train_feature', feature_loss.item(), self.step)
                    self.writer.add_scalar('Loss/step_train_smoothness_unweighted', smooth_loss_unweighted.item(),
                                           self.step)
                    self.writer.add_scalar('Metrics/step_train_keypoints', metrics.get('num_valid_keypoints', 0),
                                           self.step)

                # Visualize periodically if enabled
                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    # Detach tensors before passing to visualization
                    vis_data = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                    vis_outputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in
                                   outputs.items()}
                    try:
                        self.visualize(vis_data, vis_outputs, self.step, "train")
                    except Exception as e:
                        print(f"\nWarning: Visualization failed at step {self.step}: {e}")

                self.step += 1  # Increment global step counter

        # --- End of Epoch ---
        pbar.close()  # Close progress bar

        # Calculate average results for the epoch
        if batch_count > 0:
            avg_results = {k: v / batch_count for k, v in epoch_results.items()}
        else:
            # Handle empty epoch (no valid batches)
            print(f"Warning: {'Training' if is_training else 'Validation'} epoch {epoch + 1} had no valid batches.")
            avg_results = epoch_results  # Return zeroed results

        return avg_results

    def _log_epoch_results(self, phase, epoch, results):
        """ Logs epoch results to history dictionary and TensorBoard. """
        if not results:  # Handle empty results dict
            print(f"Warning: No results to log for {phase} epoch {epoch + 1}.")
            return

        for k, v in results.items():
            # Append result to history list
            self.history[phase][k].append(v)
            # Log to TensorBoard if writer exists
            if self.writer:
                try:
                    # Determine metric type for TensorBoard tag
                    metric_type = 'Loss' if k in self.loss_keys else 'Metrics'
                    # Log unweighted smoothness separately for scale comparison
                    if k == 'smoothness':
                        # Log the average unweighted smoothness loss for the epoch
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_smoothness_unweighted", v, epoch)
                        # Also log the weighted contribution to total loss
                        weighted_smooth = v * self.cfg.SMOOTHNESS_WEIGHT
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_smoothness_weighted", weighted_smooth, epoch)
                    elif k == 'feature':
                        # Log the feature loss (effectively weighted by 1.0)
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_feature", v, epoch)
                    elif k == 'total':
                        # Log the total loss
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_total", v, epoch)
                    else:  # Log other metrics
                        self.writer.add_scalar(f"{metric_type}_Epoch/{phase}_{k}", v, epoch)
                except Exception as e:
                    print(f"Warning: Failed to log {k}={v} to TensorBoard: {e}")

    def visualize(self, data, outputs, step, phase):
        """ Visualizes keypoints and disparity distribution for a single sample. """
        try:
            # Select the first item in the batch for visualization
            # Ensure data is on CPU and numpy format
            left_gray = data['left_gray'][0, 0].numpy()  # (H, W)

            # Get relevant outputs for the first item
            kp_left = outputs['keypoints_left'][0].numpy()  # (N, 2)
            scores_left = outputs['scores_left'][0].numpy()  # (N,)
            disparity = outputs['disparity'][0].numpy()  # (N,)

            # Filter valid keypoints based on score
            valid_mask = scores_left > 0.0  # (Use 0.0, as per v2.13)
            # --- MODIFICATION: 斑点检测器的分数是 size, 最小是 10 ---
            valid_mask = scores_left > 0.1  # (Baseline 使用 0.1, 匹配斑点大小)
            # --- END MODIFICATION ---
            kp_valid = kp_left[valid_mask]
            disp_valid = disparity[valid_mask]

            # Remove NaNs from disparity before plotting
            if len(disp_valid) > 0:
                disp_valid_finite = disp_valid[np.isfinite(disp_valid)]
            else:
                disp_valid_finite = np.array([])

            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Sparse Matching - Step: {step} ({phase})', fontsize=16)

            # --- Left Subplot: Keypoints on Image ---
            axes[0].imshow(left_gray, cmap='gray')
            if len(kp_valid) > 0:
                # Plot valid keypoints as red dots
                # --- MODIFICATION: Use smaller dots for denser visualization ---
                point_size = max(1, 10 - np.log10(len(kp_valid)))  # Dynamically smaller size
                axes[0].scatter(kp_valid[:, 0], kp_valid[:, 1], c='red', s=point_size, alpha=0.6)
                # --- END MODIFICATION ---
            axes[0].set_title(f"Detected Keypoints ({len(kp_valid)})")
            axes[0].axis('off')  # Hide axes

            # --- Right Subplot: Disparity Distribution ---
            if len(disp_valid_finite) > 0:
                # Plot histogram of finite disparity values
                # Robust range calculation
                q99 = np.percentile(disp_valid_finite, 99) if len(disp_valid_finite) > 1 else disp_valid_finite.max()
                hist_max = max(1.0, q99 * 1.1)  # Add a small margin above 99th percentile
                hist_range = (0, hist_max)
                axes[1].hist(disp_valid_finite, bins=50, range=hist_range, alpha=0.7, edgecolor='black')
                # Add line for mean disparity
                mean_disp = disp_valid_finite.mean()
                axes[1].axvline(mean_disp, color='red', linestyle='--',
                                label=f'Mean: {mean_disp:.2f}')
                axes[1].set_xlabel('Disparity (pixels)')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Disparity Distribution')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                # Display message if no valid disparities
                axes[1].set_title('Disparity Distribution (No valid disparities)')
                axes[1].text(0.5, 0.5, 'No valid disparities to plot', horizontalalignment='center',
                             verticalalignment='center', transform=axes[1].transAxes)

            # Adjust layout and save the figure
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
            save_path = os.path.join(self.vis_dir, f"{phase}_step_{step:06d}.png")
            plt.savefig(save_path, dpi=100)
            # Add figure to TensorBoard if writer exists
            if self.writer:
                # Add figure directly without reading back from file if possible
                # self.writer.add_figure(f'Visualization/{phase}', fig, step)
                # Fallback: Read image file and add
                try:
                    img = cv2.imread(save_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.writer.add_image(f'Visualization/{phase}', img, global_step=step, dataformats='HWC')
                except Exception as tb_img_e:
                    print(f"Warning: Failed to add visualization image to TensorBoard: {tb_img_e}")

            plt.close(fig)  # Close figure to free memory

        except Exception as e:
            print(f"\nWarning: Visualization failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
            if 'fig' in locals() and fig: plt.close(fig)  # Ensure figure is closed on error

    def plot_training_history(self):
        """ Plots the entire training history (losses and metrics) and saves the figure. """
        # Check if history exists and has data
        history_valid = (
                self.history and
                self.history.get('train') and self.history['train'].get('total') and
                self.history.get('val') and self.history['val'].get('total')
        )
        if not history_valid:
            print("Warning: Insufficient training history found to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sparse Matching Training History', fontsize=16)

        epochs = range(len(self.history['train']['total']))  # X-axis

        # --- Subplot [0, 0]: Total Loss ---
        try:
            train_total = [l if np.isfinite(l) else np.nan for l in self.history['train']['total']]
            val_total = [l if np.isfinite(l) else np.nan for l in self.history['val']['total']]
            axes[0, 0].plot(epochs, train_total, label='Train Loss')
            axes[0, 0].plot(epochs, val_total, label='Val Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            axes[0, 0].set_xlabel("Epochs")
            axes[0, 0].set_ylabel("Loss")
            # Adjust Y-axis limits for better visualization
            all_total_losses = [l for l in train_total + val_total if np.isfinite(l)]
            if all_total_losses:
                min_loss = min(all_total_losses)
                max_loss = max(all_total_losses)
                padding = (max_loss - min_loss) * 0.1 if (max_loss - min_loss) > 1e-6 else 0.1
                axes[0, 0].set_ylim(bottom=min_loss - padding, top=max_loss + padding)
        except Exception as e:
            print(f"Warning: Plotting Total Loss failed: {e}")
            axes[0, 0].set_title('Total Loss (Plotting Error)')

        # --- Subplot [0, 1]: Number of Valid Keypoints ---
        try:
            train_kpts = [k if np.isfinite(k) else np.nan for k in self.history['train']['num_valid_keypoints']]
            val_kpts = [k if np.isfinite(k) else np.nan for k in self.history['val']['num_valid_keypoints']]
            axes[0, 1].plot(epochs, train_kpts, label='Train Keypoints')
            axes[0, 1].plot(epochs, val_kpts, label='Val Keypoints')
            axes[0, 1].set_title('Number of Valid Keypoints (Avg per Image)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            axes[0, 1].set_xlabel("Epochs")
            axes[0, 1].set_ylabel("Count")
        except Exception as e:
            print(f"Warning: Plotting Keypoints failed: {e}")
            axes[0, 1].set_title('Number of Valid Keypoints (Plotting Error)')

        # --- Subplot [1, 0]: Mean Disparity ---
        try:
            train_disp = [d if np.isfinite(d) else np.nan for d in self.history['train']['mean_disparity']]
            val_disp = [d if np.isfinite(d) else np.nan for d in self.history['val']['mean_disparity']]
            axes[1, 0].plot(epochs, train_disp, label='Train Mean Disparity')
            axes[1, 0].plot(epochs, val_disp, label='Val Mean Disparity')
            axes[1, 0].set_title('Mean Disparity')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            axes[1, 0].set_xlabel("Epochs")
            axes[1, 0].set_ylabel("Disparity (pixels)")
        except Exception as e:
            print(f"Warning: Plotting Mean Disparity failed: {e}")
            axes[1, 0].set_title('Mean Disparity (Plotting Error)')

        # --- Subplot [1, 1]: Weighted Loss Components (Train) ---
        try:
            if self.history['train']['feature'] and self.history['train']['smoothness']:
                # Ensure values are finite before weighting/plotting
                smoothness_hist_train = [s if np.isfinite(s) else 0 for s in self.history['train']['smoothness']]
                weighted_smooth_train = [s * self.cfg.SMOOTHNESS_WEIGHT for s in smoothness_hist_train]
                feature_train = [f if np.isfinite(f) else 0 for f in self.history['train']['feature']]

                axes[1, 1].plot(epochs, feature_train, label=f'Feature Loss (W={self.cfg.PHOTOMETRIC_WEIGHT:.4f})',
                                linewidth=2)
                axes[1, 1].plot(epochs, weighted_smooth_train,
                                label=f'Smoothness Loss (W={self.cfg.SMOOTHNESS_WEIGHT:.4f})', linestyle='--')
                axes[1, 1].set_title('Weighted Loss Components (Train)')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
                axes[1, 1].set_xlabel("Epochs")
                axes[1, 1].set_ylabel("Weighted Loss Contribution")
                # Adjust Y-axis limits
                all_comp_losses = [l for l in feature_train + weighted_smooth_train if np.isfinite(l)]
                if all_comp_losses:
                    min_loss_comp = min(all_comp_losses)
                    max_loss_comp = max(all_comp_losses)
                    padding_comp = (max_loss_comp - min_loss_comp) * 0.1 if (
                                                                                    max_loss_comp - min_loss_comp) > 1e-6 else 0.01
                    ylim_bottom = max(min_loss_comp - padding_comp,
                                      -0.01) if min_loss_comp >= 0 else min_loss_comp - padding_comp
                    axes[1, 1].set_ylim(bottom=ylim_bottom, top=max_loss_comp + padding_comp)
            else:
                axes[1, 1].set_title('Weighted Loss Components (Train) (Insufficient Data)')
        except Exception as e:
            print(f"Warning: Plotting Loss Components failed: {e}")
            axes[1, 1].set_title('Weighted Loss Components (Train) (Plotting Error)')

        # --- Final Adjustments and Saving ---
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title overlap
        save_path = os.path.join(self.vis_dir, "training_history.png")
        try:
            plt.savefig(save_path)
            # print(f"Training history plot saved to {save_path}") # Less verbose
        except Exception as e:
            print(f"警告: 保存训练历史图失败: {e}")
        plt.close(fig)  # Close figure

    def update_log_file(self, epoch):
        """ Saves the current config and history to a JSON log file. """
        # Prepare log data (convert config dataclass to dict)
        log_data = {'config': asdict(self.cfg), 'epoch': epoch, 'history': self.history}
        try:
            with open(self.log_file, 'w') as f:
                # Use a custom handler for numpy floats/arrays during JSON serialization
                def json_serializer(obj):
                    if isinstance(obj, (np.float32, np.float64)):
                        # Convert numpy floats to standard Python floats
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        # Convert numpy arrays to lists
                        return obj.tolist()
                    elif isinstance(obj, (datetime, np.datetime64)):
                        # Convert datetime objects to string ISO format
                        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
                    # For other types, raise TypeError so it's caught below
                    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

                json.dump(log_data, f, indent=2, default=json_serializer)
        except TypeError as te:
            print(f"警告: JSON 序列化错误，日志文件可能不完整: {te}")
            # Optionally try saving with str() as fallback for unknown types
            try:
                with open(self.log_file + ".fallback", 'w') as f_fallback:
                    json.dump(log_data, f_fallback, indent=2, default=str)
            except Exception as fallback_e:
                print(f"Fallback log saving also failed: {fallback_e}")
        except Exception as e:
            # Catch other errors during file writing
            print(f"警告: 写入日志文件 '{self.log_file}' 失败: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Create configuration object
    cfg = Config()

    # --- MODIFICATION: 将所有启动打印移至此处 ---
    # 这样它们就不会在 num_workers > 0 的子进程中被打印
    if pynvml is None:
        print("[信息] pynvml 未安装。 无法准确显示可用显存。")
        print("       (可选) 尝试运行: pip install nvidia-ml-py")

    # --- v2.11 MODIFICATION: Hires ---
    # 我们不再自动调整，而是手动设置分辨率。
    # auto_tune_config(cfg) # <--- 禁用
    print(f"--- [v2.13 Hires Mode - Detector Fix] ---")  # v2.13 更新
    print(f"手动设置分辨率: {cfg.IMAGE_WIDTH}x{cfg.IMAGE_HEIGHT}")
    print(f"使用膨胀 (Dilation): {cfg.USE_DILATION}")

    if torch.cuda.is_available() and pynvml:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(
                f"检测到 16GB VRAM (可用: {mem_info.free / 1024 ** 2:.0f} MB / 总计: {mem_info.total / 1024 ** 2:.0f} MB)")
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"[警告] 无法通过 pynvml 获取显存: {e}")
    elif torch.cuda.is_available():
        print(f"检测到 CUDA (pynvml 未加载)")
    # --- END MODIFICATION ---

    # Sanity check for batch size
    if cfg.BATCH_SIZE <= 0:
        print("[警告] 批次大小 <= 0。 重置为 1。")
        cfg.BATCH_SIZE = 1

    # Initialize and run the Trainer
    try:
        trainer = Trainer(cfg)
        # --- MODIFICATION: 将 Trainer 的打印移至此处 ---
        print(f"[Trainer Config] 强制设置 num_workers={trainer.train_loader.num_workers} (原 Windows 默认为 0)")
        # --- END MODIFICATION ---
        trainer.train()
    except Exception as e:
        print(f"\n !!! 训练过程中发生严重错误 !!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback
        sys.exit(1)  # Exit with error code