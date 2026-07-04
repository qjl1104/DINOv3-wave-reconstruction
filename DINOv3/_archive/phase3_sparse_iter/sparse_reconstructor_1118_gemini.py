# sparse_matching_stereo.py (V5 - Photometric Loss - Stabilized)
#
# --- [ 核心诊断与 V5 修复 ] ---
#
# 诊断: V4 训练 (LR=1e-5, Smoothness=0.05) 仍然失败。
#      日志显示模型在第38个周期 "跳" 到了一个错误的视差值 (从 ~146 像素
#      跳到 ~155 像素) 并且被卡住。
#      这是因为 "精度" (SSIM Loss, W=1.0) 的力量仍然远远压倒
#      "稳定" (Smoothness Loss, W=0.05)，导致模型为了追求
#      像素完美而牺牲了几何稳定性。
#
# V5 解决方案:
# 1. [强力稳定] 将 SMOOTHNESS_WEIGHT 从 0.05 提高 10 倍到 0.5。
#    这使得 "精度" 与 "稳定" 的权重几乎达到 1:1 (1.0 vs 0.5)。
# 2. [新目标] 这将强迫模型*优先*学习一个物理上平滑、连续的波面，
#    *然后*才是在这个平滑约束下去优化像素匹配。
# 3. [保留] 保留 V4 的低学习率 (1e-5) 和 V3 的 SSIM 损失函数。
#
# ---

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

# 假设脚本在 DINOv3 目录中运行
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


@dataclass
class Config:
    """Configuration for sparse matching"""
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")
    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs_sparse_photometric_V5")  # [V5] 新的日志目录
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")

    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 100

    IMAGE_HEIGHT: int = 0
    IMAGE_WIDTH: int = 0
    MASK_THRESHOLD: int = 30
    MAX_KEYPOINTS: int = 512

    # --- [ 调试 v8/v9: 斑点检测器参数 (保留) ] ---
    BLOB_MIN_THRESHOLD: float = 30.0
    BLOB_MIN_AREA: float = 20.0
    BLOB_MAX_AREA: float = 2000.0
    BLOB_MIN_CIRCULARITY: float = 0.1
    BLOB_MIN_CONVEXITY: float = 0.90
    BLOB_MIN_INERTIA: float = 0.1
    BLOB_MAX_INERTIA: float = 0.85
    # --- [ 修改结束 ] ---

    # Matching network settings (保留)
    FEATURE_DIM: int = 768
    NUM_ATTENTION_LAYERS: int = 6
    NUM_HEADS: int = 8
    DISPARITY_CONSTRAINT_Y_THRESHOLD: int = 2
    MATCHING_TEMPERATURE: float = 10.0

    # Training settings
    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 1e-5  # [V4] 保留低学习率
    NUM_EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    GRADIENT_ACCUMULATION_STEPS: int = 1

    # Performance and Loss settings
    USE_MIXED_PRECISION: bool = True
    PHOTOMETRIC_WEIGHT: float = 1.0  # [V2] 光度损失的权重
    SMOOTHNESS_WEIGHT: float = 0.5  # [V5 最终修复] 再次提高平滑权重 (原为 0.05)，强力稳定训练。
    PATCH_SIZE_PHOTOMETRIC: int = 11  # [V2] 用于光度损失的 patch 大小 (必须为奇数)

    # Data Augmentation settings
    USE_ADVANCED_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.8

    # Early Stopping settings
    EARLY_STOPPING_PATIENCE: int = 25


def auto_tune_config(cfg: Config):
    print("[信息] auto_tune_config 已被禁用。图像分辨率将由数据集加载器根据 ROI 自动设置。")
    pass


# --- 1. [ 关键替换 ] ---
# (保留来自 v9 的 SparseKeypointDetector)
class SparseKeypointDetector(nn.Module):
    """
    使用 OpenCV 的 SimpleBlobDetector 检测稀疏关键点。
    这是从 1030 脚本移植过来的，专门用于检测泡沫。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_keypoints = cfg.MAX_KEYPOINTS

        # --- 设置 Blob 检测器 (使用 v8/v9 参数) ---
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = False
        params.minThreshold = cfg.BLOB_MIN_THRESHOLD
        params.maxThreshold = 255
        params.thresholdStep = 10

        params.filterByArea = True
        params.minArea = cfg.BLOB_MIN_AREA
        params.maxArea = cfg.BLOB_MAX_AREA

        params.filterByCircularity = True
        params.minCircularity = cfg.BLOB_MIN_CIRCULARITY

        params.filterByConvexity = True
        params.minConvexity = cfg.BLOB_MIN_CONVEXITY

        params.filterByInertia = True
        params.minInertiaRatio = cfg.BLOB_MIN_INERTIA
        params.maxInertiaRatio = cfg.BLOB_MAX_INERTIA  # <-- [v8] 使用新上限

        self.detector = cv2.SimpleBlobDetector_create(params)
        print(f"--- [检测器] 已初始化 SimpleBlobDetector (v9 修复 Bug) ---")
        print(f"    阈值: [{params.minThreshold}, {params.maxThreshold}]")
        print(f"    面积 (Area): [{params.minArea}, {params.maxArea}] (平衡远近)")
        print(f"    凸性 (Convexity): > {params.minConvexity} (主要过滤器)")
        print(f"    惯性 (Inertia): [{params.minInertiaRatio}, {params.maxInertiaRatio}] (允许椭圆和亮团)")
        print(f"-------------------------------------------------------")

    def forward(self, gray_image, mask):
        """
        Args:
            gray_image: (B, 1, H, W) - *原始* 灰度图像 (0-1 范围)
            mask: (B, 1, H, W) - (不再被此函数使用)
        Returns:
            keypoints: (B, N, 2) - (x, y) 坐标, N <= max_keypoints
            scores: (B, N) - 斑点大小 (用作分数)
        """
        B, _, H, W = gray_image.shape
        device = gray_image.device

        keypoints_list = []
        scores_list = []

        for b in range(B):
            # 转换为 numpy uint8 图像
            img_tensor = gray_image[b, 0]

            # --- [ v8 BUG 修复 ] ---
            # 直接在原始灰度图 (img_tensor) 上操作
            # BlobDetector 会使用自己的 minThreshold，不再需要外部掩码
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            # --- [ 修复结束 ] ---

            # 检测斑点
            cv_keypoints = self.detector.detect(img_np)

            if not cv_keypoints:
                keypoints_list.append(torch.zeros(1, 2, device=device))
                scores_list.append(torch.zeros(1, device=device))
                continue

            # 提取坐标和分数 (使用斑点大小作为分数)
            kp_coords = np.array([kp.pt for kp in cv_keypoints]).astype(np.float32)
            kp_scores = np.array([kp.size for kp in cv_keypoints]).astype(np.float32)

            kp_tensor = torch.from_numpy(kp_coords).to(device)
            scores_tensor = torch.from_numpy(kp_scores).to(device)

            # 按分数 (大小) 排序并限制数量
            if len(kp_tensor) > self.max_keypoints:
                sorted_indices = torch.argsort(scores_tensor, descending=True)
                kp_tensor = kp_tensor[sorted_indices[:self.max_keypoints]]
                scores_tensor = scores_tensor[sorted_indices[:self.max_keypoints]]

            keypoints_list.append(kp_tensor)
            scores_list.append(scores_tensor)

        # 填充到相同长度
        max_len = max(len(kp) for kp in keypoints_list) if keypoints_list else 0
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
                kp_pad, sc_pad = kp[:max_len], sc[:max_len]
            keypoints_padded.append(kp_pad)
            scores_padded.append(sc_pad)

        keypoints = torch.stack(keypoints_padded, dim=0)
        scores = torch.stack(scores_padded, dim=0)

        return keypoints, scores


# --- [ 替换结束 ] ---


# --- 2. DINOv3 特征提取器 (保留) ---
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
            model_path = self.cfg.DINO_LOCAL_PATH
            if not os.path.exists(model_path):
                print(f"[警告] DINO 本地路径 {model_path} 未找到。")
                print(f"       尝试从 Hugging Face Hub 加载: facebook/dinov2-base")
                model_path = 'facebook/dinov2-base'
            else:
                print(f"Loading DINO model from: {model_path}")

            return AutoModel.from_pretrained(model_path, local_files_only=os.path.exists(model_path))

        except Exception as e:
            # Exit if loading fails, as it's critical
            print(f"[FATAL] Failed loading DINOv3 model from '{model_path}': {e}")
            sys.exit(1)

    def get_feature_map(self, image):
        """
        Extracts the spatial feature map from DINOv3's last layer.
        """
        with torch.no_grad():  # Ensure no gradients are computed for DINO
            outputs = self.dino(image)
            if not hasattr(outputs, 'last_hidden_state'):
                print("[ERROR] DINO model output does not have 'last_hidden_state'. Check model compatibility.")
                sys.exit(1)

            features = outputs.last_hidden_state  # Shape (B, num_tokens, C)

        b, _, h, w = image.shape
        # Calculate the starting index of patch tokens (skip CLS and register tokens)
        start_idx = 1 + self.num_register_tokens

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
                print(f"  > 截断 token: {patch_tokens.shape[1]} -> {expected_num_patches}")
                patch_tokens = patch_tokens[:, :expected_num_patches, :]
            else:
                raise RuntimeError(
                    f"DINO token 数量不足 ({patch_tokens.shape[1]} < {expected_num_patches})，请检查图像尺寸是否为 patch_size 的整数倍。")

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
        """
        B, N, _ = keypoints.shape
        # Get the dense feature map from DINO
        feat_map = self.get_feature_map(image)  # (B, C, H_feat, W_feat)

        _, C, H_feat, W_feat = feat_map.shape
        _, _, H_img, W_img = image.shape

        if W_img <= 1 or H_img <= 1:
            print(
                f"[WARNING] Invalid image dimensions for normalization: W={W_img}, H={H_img}. Returning zero descriptors.")
            return torch.zeros((B, N, C), device=image.device)

        grid = keypoints.clone()
        grid[..., 0] = 2 * (grid[..., 0] / (W_img - 1)) - 1  # Normalize X
        grid[..., 1] = 2 * (grid[..., 1] / (H_img - 1)) - 1  # Normalize Y
        grid = grid.unsqueeze(2)

        try:
            descriptors_sampled = F.grid_sample(
                feat_map, grid,
                mode='bilinear',
                align_corners=True,
                padding_mode='border'
            )
        except Exception as e:
            print(f"[ERROR] F.grid_sample failed: {e}. Returning zero descriptors.")
            return torch.zeros((B, N, C), device=image.device)

        descriptors = descriptors_sampled.squeeze(3).permute(0, 2, 1)
        return descriptors


# --- 3. 注意力匹配网络 (保留) ---
class PositionalEncoding(nn.Module):
    """Learned 2D positional encoding for keypoints."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(2, dim)

    def forward(self, positions, image_size):
        H, W = image_size
        if W <= 1 or H <= 1:
            print(f"[WARNING] Invalid image size for PositionalEncoding: W={W}, H={H}. Returning zero encoding.")
            return torch.zeros((positions.shape[0], positions.shape[1], self.dim), device=positions.device)

        pos_normalized = positions.clone()
        pos_normalized[..., 0] = pos_normalized[..., 0] / (W - 1)
        pos_normalized[..., 1] = pos_normalized[..., 1] / (H - 1)
        return self.proj(pos_normalized)


class SelfAttentionLayer(nn.Module):
    """Standard Transformer self-attention block with pre-layer normalization."""

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
        features_pos = features + pos_enc
        qkv = self.norm1(features_pos)
        attn_out, _ = self.attn(qkv, qkv, qkv)
        features = features + attn_out
        ffn_in = self.norm2(features)
        ffn_out = self.ffn(ffn_in)
        features = features + ffn_out
        return features


class CrossAttentionLayer(nn.Module):
    """Standard Transformer cross-attention block with pre-layer normalization."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_after_res = nn.LayerNorm(dim)

    def forward(self, feat_query, feat_kv):
        q = self.norm_q(feat_query)
        kv = self.norm_kv(feat_kv)
        try:
            attn_out, attn_weights = self.attn(q, kv, kv)
        except Exception as e:
            print(f"[ERROR] Cross-Attention failed: {e}. Q shape: {q.shape}, KV shape: {kv.shape}")
            return feat_query, torch.zeros((q.shape[0], q.shape[1], kv.shape[1]), device=q.device)

        features = feat_query + attn_out
        features = self.norm_after_res(features)
        return features, attn_weights


class SparseMatchingNetwork(nn.Module):
    """
    Performs sparse feature matching using self and cross attention layers.
    (保留)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim = cfg.FEATURE_DIM
        num_layers = cfg.NUM_ATTENTION_LAYERS
        num_heads = cfg.NUM_HEADS
        print(f"Initializing SparseMatchingNetwork with {num_layers} attention layers.")

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
        feat_left = desc_left
        feat_right = desc_right
        for i, (self_l, self_r) in enumerate(zip(self.self_attn_left, self.self_attn_right)):
            feat_left = self_l(feat_left, pos_left)
            feat_right = self_r(feat_right, pos_right)
        feat_left_enhanced, _ = self.cross_attn(feat_left, feat_right)

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
        disparity_pred = torch.nan_to_num(disparity_pred, nan=0.0)
        return match_scores_constrained, disparity_pred, constraint_mask


# --- 4. [V2 核心替换] 损失函数 ---

# [V2] 移除 FeatureMetricLoss
# class FeatureMetricLoss(nn.Module):
#    ... (已移除)

# [V2] 辅助函数：从图像中采样稀疏 patch
def sample_patches(image_gray, keypoints, patch_size=11):
    """
    在 keypoints (B, N, 2) [x, y] 处从 image_gray (B, 1, H, W) 采样 patches。
    返回 (B, N, 1, patch_size, patch_size)
    """
    B, N, _ = keypoints.shape
    B, C, H, W = image_gray.shape
    device = keypoints.device

    if patch_size % 2 == 0:
        raise ValueError(f"patch_size 必须为奇数, 收到: {patch_size}")

    half_patch = patch_size // 2

    # 创建 (B, N, P*P, 2) 的采样网格
    # (P*P = patch_size * patch_size)
    P = patch_size

    # 1. 创建相对 patch 坐标 (P*P, 2)
    rel_coords_x = torch.linspace(-half_patch, half_patch, P, device=device)
    rel_coords_y = torch.linspace(-half_patch, half_patch, P, device=device)
    rel_grid_y, rel_grid_x = torch.meshgrid(rel_coords_y, rel_coords_x, indexing='ij')
    rel_grid = torch.stack((rel_grid_x, rel_grid_y), dim=-1).view(P * P, 2)  # (P*P, 2) [x, y]

    # 2. 将相对坐标添加到 keypoints
    # kp: (B, N, 1, 2)
    # rel_grid: (1, 1, P*P, 2)
    # grid_abs: (B, N, P*P, 2)
    kp_abs = keypoints.unsqueeze(2) + rel_grid.view(1, 1, P * P, 2)

    # 3. 归一化网格以用于 grid_sample
    # grid_sample 需要 (B, H_out, W_out, 2) 格式
    # 我们将其重塑为 (B, N*P, P, 2) -> F.grid_sample -> (B, C, N*P, P)
    # 然后重塑回 (B, N, C, P, P)
    grid_normalized = kp_abs.clone()
    grid_normalized[..., 0] = 2 * (grid_normalized[..., 0] / (W - 1)) - 1  # 归一化 X
    grid_normalized[..., 1] = 2 * (grid_normalized[..., 1] / (H - 1)) - 1  # 归一化 Y

    # 重塑为 grid_sample 期望的格式 (B, H_out, W_out, 2)
    # (B, N, P*P, 2) -> (B, N*P, P, 2)
    grid_normalized_reshaped = grid_normalized.view(B, N * P, P, 2)

    # 4. 采样
    patches_sampled = F.grid_sample(
        image_gray,  # (B, C, H, W)
        grid_normalized_reshaped,  # (B, N*P, P, 2)
        mode='bilinear',
        align_corners=True,  # 匹配 DINOv3 提取器
        padding_mode='border'
    )
    # patches_sampled: (B, C, N*P, P)

    # 5. 重塑回 (B, N, C, P, P)
    patches = patches_sampled.view(B, C, N, P, P).permute(0, 2, 1, 3, 4)

    return patches  # (B, N, C, P, P)


# --- [V3 核心修改] ---
# 替换 L1 损失为 SSIM 损失
# 我们需要一个可微分的 SSIM 实现
# (来源: https://github.com/Po-Hsun-Su/pytorch-ssim)
def gaussian(window_size, sigma):
    gauss = torch.exp(torch.tensor(
        [-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]
    ))
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.type())
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            # 返回每个 patch 的平均 ssim
            return ssim_map.mean(dim=[1, 2, 3])


# --- [V3 修改结束] ---


class PhotometricLoss(nn.Module):
    """
    [V2 新增]
    计算光度重建损失 (Photometric Loss) 和稀疏平滑度损失。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.PATCH_SIZE_PHOTOMETRIC
        print(f"[V2 损失] PhotometricLoss 已初始化, patch_size={self.patch_size}")
        # --- [V3 核心修改] ---
        # size_average=False 至关重要，这样我们才能得到每个patch的损失，以便进行掩码
        self.ssim = SSIM(window_size=self.patch_size, size_average=False)
        print(f"[V3 损失] 使用 SSIM 损失替换 L1 损失。")
        # --- [V3 修改结束] ---

    def forward(self, left_gray, right_gray, keypoints_left, disparity, scores_left):
        B, N_l, _ = keypoints_left.shape
        device = keypoints_left.device

        # --- 1. 光度损失 (Photometric Loss) ---

        # 1.1. 计算扭曲后的右图坐标
        # disparity (B, N_l) -> (B, N_l, 1)
        # kp_left (B, N_l, 2)
        disparity_offset = torch.stack([disparity, torch.zeros_like(disparity)], dim=-1)
        kp_right_warped = keypoints_left - disparity_offset  # (B, N_l, 2)

        # 1.2. 采样 patches
        try:
            patches_left = sample_patches(left_gray, keypoints_left, self.patch_size)
            patches_right_warped = sample_patches(right_gray, kp_right_warped, self.patch_size)
        except Exception as e:
            print(f"[错误] PhotometricLoss - 采样 patches 失败: {e}")
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # --- [V3 核心修改] ---
        # 1.3. 计算 SSIM 损失

        # SSIM 模块需要 (B*N, C, P, P) 格式
        (B, N, C, P, P) = patches_left.shape
        patches_left_reshaped = patches_left.view(B * N, C, P, P)
        patches_right_warped_reshaped = patches_right_warped.view(B * N, C, P, P)

        # ssim_val_per_patch 的形状为 (B*N,)
        ssim_val_per_patch = self.ssim(patches_left_reshaped, patches_right_warped_reshaped)

        # SSIM 是相似度 (0到1)，损失是 1 - SSIM
        photometric_loss_per_patch = 1.0 - ssim_val_per_patch

        # 1.4. 重塑为 (B, N_l) 以便掩码
        photometric_loss_per_kp = photometric_loss_per_patch.view(B, N_l)
        # --- [V3 修改结束] ---

        # --- Masking for valid keypoints ---
        # 1. 必须是检测到的点 (分数 > 0.1)
        #    'scores_left' 来自 BlobDetector，是斑点大小 (size)。
        detection_mask = (scores_left > 0.1)

        # 2. [V2] 视差必须为正
        disparity_mask = (disparity > 0.1)

        final_valid_mask = detection_mask & disparity_mask

        masked_loss = photometric_loss_per_kp * final_valid_mask
        num_valid = torch.sum(final_valid_mask)

        photometric_loss = torch.tensor(0.0, device=device)
        if num_valid > 0:
            photometric_loss = masked_loss.sum() / num_valid

        # --- 2. Smoothness Loss (Disparity Regularization) ---
        smooth_loss = self._compute_sparse_smoothness(keypoints_left, disparity, scores_left)

        return photometric_loss, smooth_loss

    def _compute_sparse_smoothness(self, keypoints, disparity, scores):
        # (此函数从旧的 FeatureMetricLoss 移至此处，功能不变)
        B, N, _ = keypoints.shape
        total_smooth_loss = 0.0
        total_valid_pairs = 0

        for b in range(B):
            kp = keypoints[b]
            disp = disparity[b]
            sc = scores[b]
            valid_mask = sc > 0.1  # 使用 Blob 分数 (大小)
            kp_valid = kp[valid_mask]
            disp_valid = disp[valid_mask]

            if len(kp_valid) < 2:
                continue

            dist = torch.cdist(kp_valid, kp_valid)
            neighbor_mask = (dist < 20) & (dist > 1e-6)  # 邻居阈值 20px
            disp_diff = (disp_valid.unsqueeze(1) - disp_valid.unsqueeze(0)).abs()
            batch_smooth_loss = (disp_diff * neighbor_mask.float()).sum()
            num_valid_pairs = neighbor_mask.sum()
            total_smooth_loss += batch_smooth_loss
            total_valid_pairs += num_valid_pairs

        smooth_loss_avg = torch.tensor(0.0, device=keypoints.device)
        if total_valid_pairs > 0:
            smooth_loss_avg = total_smooth_loss / total_valid_pairs
        return smooth_loss_avg


# --- 5. 数据集 (v8 修复) ---
# (保留来自 v9 的 RectifiedWaveStereoDataset)
class RectifiedWaveStereoDataset(Dataset):
    """
    加载, 矫正, 并 *裁剪* 图像到最小公共 ROI
    (移除了 resize)
    """

    def __init__(self, cfg: Config, is_validation=False):
        self.cfg, self.is_validation = cfg, is_validation
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"No images found in '{cfg.LEFT_IMAGE_DIR}'.")

        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            self.roi_left, self.roi_right = tuple(map(int, calib['roi_left'])), tuple(map(int, calib['roi_right']))

            # --- [ 1030 脚本的核心逻辑 ] ---
            # 自动设置 config 的分辨率为 ROI 的最小公共尺寸
            _x_l, _y_l, w_l, h_l = self.roi_left
            _x_r, _y_r, w_r, h_r = self.roi_right

            if self.cfg.IMAGE_WIDTH == 0 or self.cfg.IMAGE_HEIGHT == 0:
                print(f"[Dataset] Left ROI: {w_l}x{h_l}, Right ROI: {w_r}x{h_r}")
                target_w = min(w_l, w_r)
                target_h = min(h_l, h_r)
                print(f"[Dataset] Setting target config resolution to MINIMUM common ROI: {target_w}x{target_h}")
                self.cfg.IMAGE_WIDTH = target_w
                self.cfg.IMAGE_HEIGHT = target_h
            # --- [ 逻辑结束 ] ---

            # 如果 H/W 仍然为 0 (例如，被手动设置)，则抛出错误
            if self.cfg.IMAGE_WIDTH == 0 or self.cfg.IMAGE_HEIGHT == 0:
                raise ValueError("IMAGE_WIDTH or IMAGE_HEIGHT is zero after init.")

        except Exception as e:
            sys.exit(f"Failed to load calibration file or set ROI: {e}")

        # 训练/验证集划分 (与 1024 脚本相同)
        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]
        print(
            f"{'验证集' if is_validation else '训练集'}: {len(self.indices)} 帧, 图像尺寸: {self.cfg.IMAGE_WIDTH}x{self.cfg.IMAGE_HEIGHT}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        try:
            img_idx = self.indices[idx]
            left_path = self.left_images[img_idx]
            right_filename = 'right' + os.path.basename(left_path)[4:]
            right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_filename)

            left_raw = cv2.imread(left_path, 0)
            right_raw = cv2.imread(right_path, 0)

            if left_raw is None or right_raw is None:
                print(f"警告: 无法加载图像对 idx {idx} (file index {img_idx})")
                return None

            # 1. 矫正
            left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

            # 2. 裁剪到各自的 ROI
            x, y, w, h = self.roi_left
            left_rect_cropped = left_rect[y:y + h, x:x + w]
            x, y, w, h = self.roi_right
            right_rect_cropped = right_rect[y:y + h, x:x + w]

            # 3. [ 替换 Resize ] 裁剪到公共尺寸
            # (self.cfg.IMAGE_HEIGHT/WIDTH 已经在 __init__ 中被设为公共尺寸)
            target_h, target_w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH

            if left_rect_cropped.shape[0] < target_h or left_rect_cropped.shape[1] < target_w:
                print(f"警告: 左侧 ROI {left_rect_cropped.shape} 小于目标 {target_h, target_w} (idx {idx})")
                return None
            if right_rect_cropped.shape[0] < target_h or right_rect_cropped.shape[1] < target_w:
                print(f"警告: 右侧 ROI {right_rect_cropped.shape} 小于目标 {target_h, target_w} (idx {idx})")
                return None

            left_img = left_rect_cropped[0:target_h, 0:target_w]
            right_img = right_rect_cropped[0:target_h, 0:target_w]
            # --- [ 替换结束 ] ---

            # --- [ v8 修复 ] ---
            # 1. 从 *原始* 图像创建 mask 和 gray_tensor
            _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
            left_gray_tensor = torch.from_numpy(left_img).float().unsqueeze(0) / 255.0
            right_gray_tensor = torch.from_numpy(right_img).float().unsqueeze(0) / 255.0
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

            # 2. *然后* 才应用数据增强 (只为了 DINO)
            left_img_aug = left_img.copy()
            right_img_aug = right_img.copy()

            if not self.is_validation and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
                if np.random.rand() < 0.5:
                    brightness = np.random.uniform(0.7, 1.3)
                    left_img_aug = np.clip(left_img_aug * brightness, 0, 255).astype(np.uint8)
                    right_img_aug = np.clip(right_img_aug * brightness, 0, 255).astype(np.uint8)
                if np.random.rand() < 0.5:
                    contrast = np.random.uniform(0.7, 1.3)
                    mean_l, mean_r = left_img_aug.mean(), right_img_aug.mean()
                    left_img_aug = np.clip((left_img_aug - mean_l) * contrast + mean_l, 0, 255).astype(np.uint8)
                    right_img_aug = np.clip((right_img_aug - mean_r) * contrast + mean_r, 0, 255).astype(np.uint8)

            # 3. 从 *增强后* 的图像创建 rgb_tensor
            left_rgb_tensor = torch.from_numpy(
                cv2.cvtColor(left_img_aug, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0
            right_rgb_tensor = torch.from_numpy(
                cv2.cvtColor(right_img_aug, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0
            # --- [ 修复结束 ] ---

            return {
                'left_gray': left_gray_tensor,  # (来自原始图, 用于 Detector 和 PhotometricLoss)
                'right_gray': right_gray_tensor,  # (来自原始图, 用于 Detector 和 PhotometricLoss)
                'left_rgb': left_rgb_tensor,  # (来自增强图, 用于 DINO)
                'right_rgb': right_rgb_tensor,  # (来自增强图, 用于 DINO)
                'mask': mask_tensor  # (来自原始图, 不再用于 v2 loss)
            }
        except Exception as e:
            print(f"Warning at idx {idx}: {e}")
            import traceback
            traceback.print_exc()
            return None


# --- [ 替换结束 ] ---


# --- 6. 完整模型 (保留 v9) ---
class SparseMatchingStereoModel(nn.Module):
    """ The complete sparse matching stereo model pipeline. """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # 初始化子模块
        self.keypoint_detector = SparseKeypointDetector(cfg)  # <-- 现在是 Blob 检测器
        self.feature_extractor = DINOv3FeatureExtractor(cfg)
        self.matcher = SparseMatchingNetwork(cfg)

        print(f"Sparse Matching Model: max {cfg.MAX_KEYPOINTS} keypoints, "
              f"{cfg.NUM_ATTENTION_LAYERS} attention layers")
        print(f"[V2] Using NEW Photometric Loss (SSIM on patches)")  # [V3] 更新日志

    def forward(self, left_gray, right_gray, left_rgb, right_rgb, mask):
        B, _, H, W = left_gray.shape

        # 1. 检测稀疏关键点
        # [v8] 'mask' 被传入, 但 detector (v8) 会 *忽略* 它, 只使用 left_gray
        kp_left, scores_left = self.keypoint_detector(left_gray, mask)
        # (右图不使用 mask)
        kp_right, scores_right = self.keypoint_detector(right_gray, torch.ones_like(right_gray))

        # 2. 提取 DINO 特征
        # [v8] 'left_rgb' 是增强过的, 'kp_left' 是来自原始图的
        desc_left = self.feature_extractor(left_rgb, kp_left)
        desc_right = self.feature_extractor(right_rgb, kp_right)

        # 3. 匹配
        match_scores, disparity, constraint_mask = self.matcher(
            desc_left, desc_right, kp_left, kp_right, (H, W)
        )

        # [ V2 ]
        # 返回 loss_fn 需要的所有内容
        return {
            'keypoints_left': kp_left,
            'scores_left': scores_left,
            'disparity': disparity,
            # (以下用于评估和可视化)
            'keypoints_right': kp_right,
            'scores_right': scores_right,
            'descriptors_left': desc_left,
            'descriptors_right': desc_right,
            'match_scores': match_scores,
            'constraint_mask': constraint_mask
        }


# --- 7. 评估 (保留 v9) ---
class EvaluationMetrics:
    @staticmethod
    def compute_sparse_metrics(disparity, keypoints_left, scores_left):
        metrics = {}
        B, N = disparity.shape
        valid_mask = scores_left > 0.1
        disparity_flat = disparity[valid_mask]

        if disparity_flat.numel() > 0:
            finite_disp = disparity_flat[torch.isfinite(disparity_flat)]
            if finite_disp.numel() > 0:
                metrics['mean_disparity'] = finite_disp.mean().item()
                metrics['std_disparity'] = finite_disp.std().item()
            else:
                metrics['mean_disparity'] = 0.0
                metrics['std_disparity'] = 0.0
            metrics['num_valid_keypoints'] = valid_mask.sum().item() / B
        else:
            metrics['mean_disparity'] = 0.0
            metrics['std_disparity'] = 0.0
            metrics['num_valid_keypoints'] = 0
        return metrics


# --- Collate 函数 (保留 v9) ---
def collate_fn(batch):
    """Custom collate function to handle None samples"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# --- 8. 训练器 (Trainer, [V2] 更新了 loss) ---
class Trainer:
    """ Handles the training and validation loops, logging, visualization, and checkpointing. """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR, self.timestamp)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.vis_dir = os.path.join(self.run_dir, "visualizations")
        self.log_dir_json = os.path.join(self.run_dir, "logs")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")
        for d in [self.ckpt_dir, self.vis_dir, self.log_dir_json, self.tb_dir]:
            os.makedirs(d, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # [修改] Dataset 现在会自动设置 cfg.IMAGE_WIDTH/HEIGHT
        train_ds = RectifiedWaveStereoDataset(cfg, is_validation=False)
        val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)
        # [修改结束]

        num_workers = 0 if sys.platform == 'win32' else 4
        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_fn, num_workers=num_workers,
                                       pin_memory=True if self.device.type == 'cuda' else False)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=num_workers,
                                     pin_memory=True if self.device.type == 'cuda' else False)

        try:
            self.writer = SummaryWriter(log_dir=self.tb_dir) if SummaryWriter else None
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard SummaryWriter: {e}")
            self.writer = None

        self.model = SparseMatchingStereoModel(cfg).to(self.device)
        self.loss_fn = PhotometricLoss(cfg)  # [V2] 切换到新的损失函数
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=cfg.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-7
        )
        self.evaluator = EvaluationMetrics()
        self.scaler = torch.amp.GradScaler(enabled=cfg.USE_MIXED_PRECISION and self.device.type == 'cuda')
        self.step = 0
        self.log_file = os.path.join(self.log_dir_json, "training_log.json")

        self.loss_keys = ['total', 'photometric', 'smoothness']  # [V2] 'feature' -> 'photometric'
        self.metric_keys = ['mean_disparity', 'std_disparity', 'num_valid_keypoints']
        self.history = {
            'train': {k: [] for k in self.loss_keys + self.metric_keys},
            'val': {k: [] for k in self.loss_keys + self.metric_keys}
        }
        self.update_log_file(-1)

        # [V2] 保存配置副本
        try:
            with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
                json.dump(asdict(self.cfg), f, indent=2)
        except Exception as e:
            print(f"警告: 无法保存 config.json: {e}")

    def train(self):
        print("\n--- [V5 训练] Starting Sparse Matching Training (BlobDetector + SSIM + Smooth=0.5) ---")  # [V5]
        print(
            f"Config: Max keypoints: {self.cfg.MAX_KEYPOINTS}, Layers: {self.cfg.NUM_ATTENTION_LAYERS}, LR: {self.cfg.LEARNING_RATE:.1e}, Smooth Weight: {self.cfg.SMOOTHNESS_WEIGHT:.1e}")
        best_val_loss = float('inf')
        epochs_no_improve = 0

        try:
            for epoch in range(self.cfg.NUM_EPOCHS):
                train_results = self._run_epoch(epoch, is_training=True)
                if not train_results:
                    print(f"Epoch {epoch + 1}: Training epoch failed or returned empty results. Stopping.")
                    break
                self._log_epoch_results('train', epoch, train_results)

                with torch.no_grad():
                    val_results = self._run_epoch(epoch, is_training=False)
                if not val_results:
                    print(f"Epoch {epoch + 1}: Validation epoch failed or returned empty results. Stopping.")
                    break
                self._log_epoch_results('val', epoch, val_results)

                current_val_loss = val_results.get('total', float('inf'))

                print(  # [V2] 'feat' -> 'photo'
                    f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> "
                    f"Train Loss: {train_results.get('total', float('nan')):.4f} (photo: {train_results.get('photometric', float('nan')):.4f}) | "
                    f"Val Loss: {current_val_loss:.4f} (photo: {val_results.get('photometric', float('nan')):.4f}) | "
                    f"Val Keypoints: {val_results.get('num_valid_keypoints', 0):.0f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

                self.update_log_file(epoch)
                if self.cfg.VISUALIZE_TRAINING:
                    try:
                        self.plot_training_history()
                    except Exception as plot_e:
                        print(f"Warning: Failed to plot training history for epoch {epoch + 1}: {plot_e}")

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    epochs_no_improve = 0
                    save_path = os.path.join(self.ckpt_dir, "best_model_sparse.pth")
                    try:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"  Val loss improved to {best_val_loss:.4f}. Model saved to {save_path}")
                    except Exception as save_e:
                        print(f"Warning: Failed to save model checkpoint: {save_e}")
                else:
                    epochs_no_improve += 1
                    print(f"  No validation loss improvement for {epochs_no_improve} epochs.")
                    if epochs_no_improve >= self.cfg.EARLY_STOPPING_PATIENCE:
                        print(
                            f"--- Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss for {self.cfg.EARLY_STOPPING_PATIENCE} epochs. ---")
                        break
                self.scheduler.step()
        except KeyboardInterrupt:
            print("\n--- Training interrupted by user (KeyboardInterrupt). ---")
        except Exception as train_e:
            print(f"\n--- Training failed due to an error: {train_e} ---")
            import traceback
            traceback.print_exc()
        finally:
            print("--- Training loop finished or stopped. ---")
            if self.writer:
                print("Closing TensorBoard writer...")
                self.writer.close()
            print("--- Training script exiting. ---")

    def _pad_inputs(self, *tensors):
        """ Pads input tensors height and width to be divisible by the patch size (16 for DINO). """
        if not tensors: return []
        # [修改] 确保第一个张量不是 None
        if tensors[0] is None:
            print("[警告] _pad_inputs: 第一个张量为 None。")
            return tensors

        b, c, h, w = tensors[0].shape
        patch_size = 16
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        if pad_h > 0 or pad_w > 0:
            padded_tensors = []
            for x in tensors:
                if x is not None:
                    padded_tensors.append(F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0))
                else:
                    padded_tensors.append(None)
            return padded_tensors
        return list(tensors)

    def _run_epoch(self, epoch, is_training):
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [{'Train' if is_training else 'Val'}]",
                    leave=True)
        epoch_results = {k: 0.0 for k in self.loss_keys + self.metric_keys}
        batch_count = 0

        for data in pbar:
            if data is None:
                print("Warning: Skipping None batch returned by collate_fn.")
                continue

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

            try:
                left_gray, right_gray, left_rgb, right_rgb, mask = self._pad_inputs(
                    left_gray, right_gray, left_rgb, right_rgb, mask
                )
            except Exception as e:
                print(f"Error padding inputs: {e}. Skipping batch.")
                continue

            autocast_kwargs = {'device_type': self.device.type, 'enabled': self.cfg.USE_MIXED_PRECISION}

            with torch.autocast(**autocast_kwargs):
                try:
                    outputs = self.model(left_gray, right_gray, left_rgb, right_rgb, mask)
                    if outputs is None:
                        raise ValueError("Model forward pass returned None")
                except Exception as e:
                    print(f"\nError during model forward pass: {e}. Skipping batch.")
                    if is_training: self.optimizer.zero_grad()
                    continue

                try:
                    # [ V2 修复 ]
                    # 调用新的 PhotometricLoss
                    photometric_loss, smooth_loss_unweighted = self.loss_fn(
                        left_gray,  # [V2] 传入原始灰度图
                        right_gray,  # [V2] 传入原始灰度图
                        outputs['keypoints_left'],
                        outputs['disparity'],
                        outputs['scores_left']
                    )
                    total_loss = (self.cfg.PHOTOMETRIC_WEIGHT * photometric_loss +
                                  self.cfg.SMOOTHNESS_WEIGHT * smooth_loss_unweighted)
                    if not torch.isfinite(total_loss):
                        raise ValueError("Loss became NaN or Inf during calculation.")
                except KeyError as e:
                    print(f"\nError: Missing key {e} in model outputs during loss calculation. Skipping batch.")
                    if is_training: self.optimizer.zero_grad()
                    continue
                except Exception as e:
                    print(f"\nError calculating loss: {e}. Skipping batch.")
                    if is_training: self.optimizer.zero_grad()
                    continue

            if is_training:
                accum_steps = self.cfg.GRADIENT_ACCUMULATION_STEPS
                scaled_loss = self.scaler.scale(total_loss / accum_steps)
                try:
                    scaled_loss.backward()
                    if (self.step + 1) % accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                except Exception as e:
                    print(f"\nError during backpropagation or optimizer step: {e}. Skipping step.")
                    self.optimizer.zero_grad()

            try:
                metrics = self.evaluator.compute_sparse_metrics(
                    outputs['disparity'].detach(),
                    outputs['keypoints_left'].detach(),
                    outputs['scores_left'].detach()
                )
            except Exception as e:
                print(f"\nWarning: Failed to compute metrics: {e}. Metrics will be zero.")
                metrics = {k: 0.0 for k in self.metric_keys}

            epoch_results['total'] += total_loss.item()
            epoch_results['photometric'] += photometric_loss.item()  # [V2]
            epoch_results['smoothness'] += smooth_loss_unweighted.item()
            for k in self.metric_keys:
                epoch_results[k] += metrics.get(k, 0.0)
            batch_count += 1
            pbar.set_postfix({
                'loss': total_loss.item(),
                'photo_loss': photometric_loss.item(),  # [V2]
                'kpts': metrics.get('num_valid_keypoints', 0),
                'disp': metrics.get('mean_disparity', 0.0)
            })

            if is_training:
                if self.writer:
                    self.writer.add_scalar('Loss/step_train_total', total_loss.item(), self.step)
                    self.writer.add_scalar('Loss/step_train_photometric', photometric_loss.item(), self.step)  # [V2]
                    self.writer.add_scalar('Loss/step_train_smoothness_unweighted', smooth_loss_unweighted.item(),
                                           self.step)
                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    vis_data = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                    vis_outputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in
                                   outputs.items()}
                    try:
                        self.visualize(vis_data, vis_outputs, self.step, "train")
                    except Exception as e:
                        print(f"\nWarning: Visualization failed at step {self.step}: {e}")
                self.step += 1

        pbar.close()
        if batch_count > 0:
            avg_results = {k: v / batch_count for k, v in epoch_results.items()}
        else:
            print(f"Warning: {'Training' if is_training else 'Validation'} epoch {epoch + 1} had no valid batches.")
            avg_results = epoch_results
        return avg_results

    def _log_epoch_results(self, phase, epoch, results):
        if not results:
            print(f"Warning: No results to log for {phase} epoch {epoch + 1}.")
            return
        for k, v in results.items():
            self.history[phase][k].append(v)
            if self.writer:
                try:
                    metric_type = 'Loss' if k in self.loss_keys else 'Metrics'
                    if k == 'smoothness':
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_smoothness_unweighted", v, epoch)
                        weighted_smooth = v * self.cfg.SMOOTHNESS_WEIGHT
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_smoothness_weighted", weighted_smooth, epoch)
                    elif k == 'photometric':  # [V2]
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_photometric", v, epoch)
                    elif k == 'total':
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_total", v, epoch)
                    else:
                        self.writer.add_scalar(f"{metric_type}_Epoch/{phase}_{k}", v, epoch)
                except Exception as e:
                    print(f"Warning: Failed to log {k}={v} to TensorBoard: {e}")

    def visualize(self, data, outputs, step, phase):
        """ Visualizes keypoints and disparity distribution for a single sample. """
        try:
            # [v8] 修复: 'left_gray' 现在是未经增强的原始图
            left_gray = data['left_gray'][0, 0].numpy()
            kp_left = outputs['keypoints_left'][0].numpy()
            scores_left = outputs['scores_left'][0].numpy()
            disparity = outputs['disparity'][0].numpy()

            valid_mask = scores_left > 0.1  # 使用 Blob 分数
            kp_valid = kp_left[valid_mask]
            disp_valid = disparity[valid_mask]

            if len(disp_valid) > 0:
                disp_valid_finite = disp_valid[np.isfinite(disp_valid)]
            else:
                disp_valid_finite = np.array([])

            # --- [ v9 修复: 提高预览图分辨率 ] ---
            # 1. 计算图像的宽高比
            img_h, img_w = left_gray.shape
            aspect_ratio = img_w / img_h

            # 2. 定义一个基础高度 (例如 10 英寸)，并根据宽高比计算宽度
            fig_h = 10
            fig_w = fig_h * aspect_ratio * 2  # *2 因为有两个子图

            # 3. 确保宽度足够大 (至少 30 英寸)
            if fig_w < 30: fig_w = 30

            fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
            fig.suptitle(f'Sparse Matching (Photometric Loss) - Step: {step} ({phase})', fontsize=16)  # [V2]
            # --- [ v9 修复结束 ] ---

            # --- 左图: 关键点 ---
            axes[0].imshow(left_gray, cmap='gray')
            if len(kp_valid) > 0:
                axes[0].scatter(kp_valid[:, 0], kp_valid[:, 1], c='red', s=10, alpha=0.6)
            axes[0].set_title(f"Detected Keypoints ({len(kp_valid)})")
            axes[0].axis('off')

            # --- 右图: 视差分布 ---
            if len(disp_valid_finite) > 0:
                q99 = np.percentile(disp_valid_finite, 99) if len(disp_valid_finite) > 1 else disp_valid_finite.max()
                hist_max = max(1.0, q99 * 1.1)
                hist_range = (0, hist_max)
                axes[1].hist(disp_valid_finite, bins=50, range=hist_range, alpha=0.7, edgecolor='black')
                mean_disp = disp_valid_finite.mean()
                axes[1].axvline(mean_disp, color='red', linestyle='--',
                                label=f'Mean: {mean_disp:.2f}')
                axes[1].set_xlabel('Disparity (pixels)')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Disparity Distribution')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].set_title('Disparity Distribution (No valid disparities)')
                axes[1].text(0.5, 0.5, 'No valid disparities to plot', horizontalalignment='center',
                             verticalalignment='center', transform=axes[1].transAxes)

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # --- [ v9 修复: 提高预览图分辨率 ] ---
            # 使用更高的 DPI 保存图像
            save_path = os.path.join(self.vis_dir, f"{phase}_step_{step:06d}.png")
            plt.savefig(save_path, dpi=150)  # (原为 100)
            # --- [ v9 修复结束 ] ---

            if self.writer:
                try:
                    img = cv2.imread(save_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.writer.add_image(f'Visualization/{phase}', img, global_step=step, dataformats='HWC')
                except Exception as tb_img_e:
                    print(f"Warning: Failed to add visualization image to TensorBoard: {tb_img_e}")
            plt.close(fig)
        except Exception as e:
            print(f"\nWarning: Visualization failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
            if 'fig' in locals() and fig: plt.close(fig)

    def plot_training_history(self):
        """ Plots the entire training history (losses and metrics) and saves the figure. """
        history_valid = (
                self.history and
                self.history.get('train') and self.history['train'].get('total') and
                self.history.get('val') and self.history['val'].get('total')
        )
        if not history_valid:
            print("Warning: Insufficient training history found to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sparse Matching Training History (Photometric Loss)', fontsize=16)  # [V2]

        epochs = range(len(self.history['train']['total']))

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
            all_total_losses = [l for l in train_total + val_total if np.isfinite(l)]
            if all_total_losses:
                min_loss = min(all_total_losses)
                max_loss = max(all_total_losses)
                padding = (max_loss - min_loss) * 0.1 if (max_loss - min_loss) > 1e-6 else 0.1
                axes[0, 0].set_ylim(bottom=min_loss - padding, top=max_loss + padding)
        except Exception as e:
            print(f"Warning: Plotting Total Loss failed: {e}")
            axes[0, 0].set_title('Total Loss (Plotting Error)')

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

        try:
            if self.history['train']['photometric'] and self.history['train']['smoothness']:  # [V2]
                smoothness_hist_train = [s if np.isfinite(s) else 0 for s in self.history['train']['smoothness']]
                weighted_smooth_train = [s * self.cfg.SMOOTHNESS_WEIGHT for s in smoothness_hist_train]
                photometric_train = [f if np.isfinite(f) else 0 for f in self.history['train']['photometric']]  # [V2]
                axes[1, 1].plot(epochs, photometric_train,
                                label=f'Photometric Loss (W={self.cfg.PHOTOMETRIC_WEIGHT:.4f})',  # [V2]
                                linewidth=2)
                axes[1, 1].plot(epochs, weighted_smooth_train,
                                label=f'Smoothness Loss (W={self.cfg.SMOOTHNESS_WEIGHT:.4f})', linestyle='--')
                axes[1, 1].set_title('Weighted Loss Components (Train)')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
                axes[1, 1].set_xlabel("Epochs")
                axes[1, 1].set_ylabel("Weighted Loss Contribution")
                all_comp_losses = [l for l in photometric_train + weighted_smooth_train if np.isfinite(l)]  # [V2]
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

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.vis_dir, "training_history.png")
        try:
            plt.savefig(save_path)
        except Exception as e:
            print(f"警告: 保存训练历史图失败: {e}")
        plt.close(fig)

    def update_log_file(self, epoch):
        log_data = {'config': asdict(self.cfg), 'epoch': epoch, 'history': self.history}
        try:
            with open(self.log_file, 'w') as f:
                def json_serializer(obj):
                    if isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (datetime, np.datetime64)):
                        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
                    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

                json.dump(log_data, f, indent=2, default=json_serializer)
        except TypeError as te:
            print(f"警告: JSON 序列化错误，日志文件可能不完整: {te}")
            try:
                with open(self.log_file + ".fallback", 'w') as f_fallback:
                    json.dump(log_data, f_fallback, indent=2, default=str)
            except Exception as fallback_e:
                print(f"Fallback log saving also failed: {fallback_e}")
        except Exception as e:
            print(f"警告: 写入日志文件 '{self.log_file}' 失败: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. 创建配置
    cfg = Config()

    # 2. 移除 auto_tune_config
    # auto_tune_config(cfg) # (禁用)

    # 3. 检查 Batch Size (保持不变)
    if cfg.BATCH_SIZE <= 0:
        print("[警告] 批次大小 <= 0。 重置为 1。")
        cfg.BATCH_SIZE = 1

    # 4. 初始化并运行训练器
    try:
        trainer = Trainer(cfg)
        trainer.train()
    except Exception as e:
        print(f"\n !!! 训练过程中发生严重错误 !!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)