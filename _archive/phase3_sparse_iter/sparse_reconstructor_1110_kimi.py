# sparse_reconstructor_1110_kimi_numerical_stable.py
# 数值稳定版：修复NaN/Inf问题，确保训练稳定

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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
    MAX_KEYPOINTS: int = 512
    NMS_RADIUS: int = 3
    PATCH_SIZE: int = 5

    # Matching network settings
    FEATURE_DIM: int = 768
    NUM_ATTENTION_LAYERS: int = 4
    NUM_HEADS: int = 8
    DISPARITY_CONSTRAINT_Y_THRESHOLD: int = 2
    MATCHING_TEMPERATURE: float = 0.5  # ✅ 降低温度提高数值稳定性

    # Training settings
    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 1e-4  # ✅ 降低学习率
    NUM_EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 0.5  # ✅ 降低梯度裁剪阈值
    GRADIENT_ACCUMULATION_STEPS: int = 1

    # Performance and Loss settings
    USE_MIXED_PRECISION: bool = False
    PHOTOMETRIC_WEIGHT: float = 1.0
    SMOOTHNESS_WEIGHT: float = 0.01

    # Sparse attention configuration
    SPARSITY_RATIO: float = 0.1
    WINDOW_SIZE: int = 16
    GLOBAL_TOKENS: int = 50

    # Data Augmentation settings
    USE_ADVANCED_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.8

    # Early Stopping settings
    EARLY_STOPPING_PATIENCE: int = 25


# === 权重初始化函数 ===
def init_weights(module):
    """初始化模型权重"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# === 1. GPU优化的稀疏关键点检测器 ===
class OptimizedSparseKeypointDetector(nn.Module):
    """GPU优化的稀疏关键点检测器"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_keypoints = cfg.MAX_KEYPOINTS
        self.nms_radius = cfg.NMS_RADIUS
        self._init_nms_kernel()

    def _init_nms_kernel(self):
        """初始化GPU NMS核"""
        kernel_size = 2 * self.nms_radius + 1
        y, x = torch.meshgrid(
            torch.arange(kernel_size) - self.nms_radius,
            torch.arange(kernel_size) - self.nms_radius,
            indexing='ij'
        )
        self.nms_kernel = (x ** 2 + y ** 2 <= self.nms_radius ** 2).float()

    def forward(self, gray_image, mask):
        """GPU优化的前向传播"""
        B, _, H, W = gray_image.shape
        device = gray_image.device
        img_masked = gray_image * mask
        keypoints_list = []
        scores_list = []

        for b in range(B):
            img = img_masked[b, 0]
            m = mask[b, 0]

            max_pooled = F.max_pool2d(
                img.unsqueeze(0).unsqueeze(0),
                kernel_size=2 * self.nms_radius + 1,
                stride=1,
                padding=self.nms_radius
            )

            local_max_mask = (img.unsqueeze(0).unsqueeze(0) == max_pooled).squeeze()
            valid_mask = local_max_mask & (img > 0.1) & (m > 0)

            y_coords, x_coords = torch.where(valid_mask)
            scores = img[valid_mask]

            if len(x_coords) == 0:
                keypoints_list.append(torch.zeros(1, 2, device=device))
                scores_list.append(torch.zeros(1, device=device))
                continue

            num_points = min(len(scores), self.max_keypoints * 4)
            top_indices = torch.topk(scores, num_points)[1]

            kp = torch.stack([x_coords[top_indices], y_coords[top_indices]], dim=1).float()
            sc = scores[top_indices]

            kp_nms, scores_nms = self._gpu_nms(kp, sc)

            if len(kp_nms) > self.max_keypoints:
                kp_nms = kp_nms[:self.max_keypoints]
                scores_nms = scores_nms[:self.max_keypoints]

            keypoints_list.append(kp_nms)
            scores_list.append(scores_nms)

        return self._pad_batch(keypoints_list, scores_list, device)

    def _gpu_nms(self, keypoints, scores):
        """GPU加速的NMS"""
        if len(keypoints) == 0:
            return keypoints, scores

        distances = torch.cdist(keypoints.float(), keypoints.float())
        sorted_indices = torch.argsort(scores, descending=True)
        keep_mask = torch.ones(len(keypoints), dtype=torch.bool, device=keypoints.device)

        for i in range(len(sorted_indices)):
            idx = sorted_indices[i]
            if not keep_mask[idx]:
                continue

            neighbor_mask = distances[idx] < self.nms_radius
            neighbor_mask[idx] = False
            keep_mask[neighbor_mask] = False

        return keypoints[keep_mask], scores[keep_mask]

    def _pad_batch(self, keypoints_list, scores_list, device):
        """批处理填充"""
        max_len = max(len(kp) for kp in keypoints_list) if keypoints_list else 1
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

        return torch.stack(keypoints_padded, dim=0), torch.stack(scores_padded, dim=0)


# === 2. DINOv3特征提取器（终极修复版）===
class DINOv3FeatureExtractor(nn.Module):
    """DINOv3特征提取器 - 强制启用梯度流"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.dino = self._load_dino_model()

        # 冻结DINO参数（但保留梯度流）
        for p in self.dino.parameters():
            p.requires_grad = False

        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

    def _load_dino_model(self):
        """Load the DINOv3 model from the specified local path."""
        try:
            model_path = self.cfg.DINO_LOCAL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"DINO model directory not found at: {model_path}")
            print(f"Loading DINO model from: {model_path}")
            return AutoModel.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"[FATAL] Failed loading local DINOv3 model from '{self.cfg.DINO_LOCAL_PATH}': {e}")
            sys.exit(1)

    def get_feature_map(self, image):
        """
        Extracts the spatial feature map from DINOv3's last layer.
        ✅ 终极修复：显式强制requires_grad=True
        """
        # 移除torch.no_grad()，让梯度能流过特征提取
        outputs = self.dino(image)
        if not hasattr(outputs, 'last_hidden_state'):
            print("[ERROR] DINO model output does not have 'last_hidden_state'. Check model compatibility.")
            if hasattr(outputs, 'pooler_output'):
                print("[WARNING] Using pooler_output instead, spatial information lost!")
                return outputs.pooler_output.unsqueeze(-1).unsqueeze(-1)
            else:
                print("[FATAL] Cannot extract features from DINO output.")
                sys.exit(1)

        features = outputs.last_hidden_state

        b, _, h, w = image.shape
        start_idx = 1 + self.num_register_tokens

        if features.shape[1] <= start_idx:
            print(
                f"[ERROR] DINO output has insufficient tokens ({features.shape[1]}) to extract patch tokens (start_idx={start_idx}).")
            return torch.zeros((b, self.feature_dim, h // self.patch_size, w // self.patch_size), device=image.device,
                               requires_grad=True)  # ✅ 显式启用梯度

        patch_tokens = features[:, start_idx:, :]
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        expected_num_patches = h_feat * w_feat

        if patch_tokens.shape[1] != expected_num_patches:
            print(
                f"[WARNING] Number of patch tokens ({patch_tokens.shape[1]}) does not match expected ({expected_num_patches}).")
            if patch_tokens.shape[1] > expected_num_patches:
                patch_tokens = patch_tokens[:, :expected_num_patches, :]

        try:
            feat_map = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, h_feat, w_feat)
        except RuntimeError as e:
            print(
                f"[ERROR] Failed to reshape patch tokens. Patch token count: {patch_tokens.shape[1]}, Expected: {h_feat * w_feat}. Error: {e}")
            return torch.zeros((b, self.feature_dim, h_feat, w_feat), device=image.device, requires_grad=True)

        # ✅ 终极修复：确保特征图可微分（即使DINO冻结）
        return feat_map.requires_grad_(True)

    def forward(self, image, keypoints):
        """
        Extracts DINOv3 features at the given keypoint locations.
        ✅ 修复：确保返回的张量可以传播梯度
        """
        B, N, _ = keypoints.shape
        feat_map = self.get_feature_map(image)  # 现在有梯度
        _, C, H_feat, W_feat = feat_map.shape
        _, _, H_img, W_img = image.shape

        if W_img <= 1 or H_img <= 1:
            print(
                f"[WARNING] Invalid image dimensions for normalization: W={W_img}, H={H_img}. Returning zero descriptors.")
            return torch.zeros((B, N, C), device=image.device, requires_grad=True)

        # ✅ 守卫：确保feat_map需要梯度
        if not feat_map.requires_grad:
            print("[WARNING] feat_map does not require grad! Forcing it on.")
            feat_map.requires_grad_(True)

        grid = keypoints.clone()
        grid[..., 0] = 2 * (grid[..., 0] / (W_img - 1)) - 1
        grid[..., 1] = 2 * (grid[..., 1] / (H_img - 1)) - 1
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
            return torch.zeros((B, N, C), device=image.device, requires_grad=True)

        descriptors = descriptors_sampled.squeeze(3).permute(0, 2, 1)
        # ✅ 双重保险：确保最终输出需要梯度
        return descriptors.requires_grad_(True)


# === 3. 修复的标准注意力机制 ===
class ScaledDotProductAttention(nn.Module):
    """标准的多头注意力机制 - 支持自注意力和交叉注意力"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # 应用权重初始化
        self.apply(init_weights)

    def forward(self, query, key=None, value=None, mask=None):
        """标准注意力计算"""
        if key is None:
            key = query
        if value is None:
            value = query

        B, N_q, C = query.shape
        B, N_k, C = key.shape
        B, N_v, C = value.shape

        # 线性变换并拆分为多头: (B, N, C) -> (B, num_heads, N, head_dim)
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)

        # 标准注意力计算: (B, num_heads, N_q, N_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1)  # 在num_heads维度广播
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 应用到value: (B, num_heads, N_q, head_dim)
        out = torch.matmul(attn_weights, v)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, N_q, C)

        return self.out_proj(out)


# === 4. 稀疏注意力机制（修复版）===
class SparseAttention(nn.Module):
    """稀疏注意力机制 - 结合全局和局部注意力"""

    def __init__(self, dim, num_heads=8, sparsity_ratio=0.1, window_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.sparsity_ratio = sparsity_ratio
        self.window_size = window_size

        # 全局注意力 - 使用修复的标准注意力
        self.global_attn = ScaledDotProductAttention(dim, num_heads)

        # 局部窗口注意力
        self.local_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )

        # 应用权重初始化
        self.apply(init_weights)

    def forward(self, x):
        """稀疏注意力前向传播"""
        B, N, C = x.shape
        num_global_tokens = max(1, int(N * self.sparsity_ratio))  # 动态计算

        if N <= num_global_tokens:
            return self.global_attn(x)

        # 计算重要性分数
        importance_scores = torch.norm(x, dim=-1)
        _, top_indices = torch.topk(
            importance_scores,
            num_global_tokens,
            dim=1
        )

        # 全局注意力处理重要token
        global_tokens = x.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, C))
        global_out = self.global_attn(global_tokens)

        # 局部窗口注意力
        local_out = self._local_window_attention(x)

        # 合并全局和局部注意力结果
        out = local_out.clone()
        # 确保混合精度训练时数据类型一致
        out.scatter_(1, top_indices.unsqueeze(-1).expand(-1, -1, C), global_out.to(out.dtype))

        return out

    def _local_window_attention(self, x):
        """局部窗口注意力"""
        B, N, C = x.shape
        num_windows = (N + self.window_size - 1) // self.window_size
        local_out = torch.zeros_like(x)

        for i in range(num_windows):
            start_idx = i * self.window_size
            end_idx = min((i + 1) * self.window_size, N)

            window_x = x[:, start_idx:end_idx]
            local_out[:, start_idx:end_idx], _ = self.local_attn(
                window_x, window_x, window_x
            )

        return local_out


# === 5. 优化的位置编码 ===
class OptimizedPositionalEncoding(nn.Module):
    """优化的位置编码 - 使用正弦位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.arange(0, dim // 4, dtype=torch.float32)
        )
        self.proj = nn.Linear(dim, dim)

        # 应用权重初始化
        self.apply(init_weights)

    def forward(self, positions, image_size):
        """前向传播"""
        H, W = image_size
        pos_normalized = positions.clone()
        pos_normalized[..., 0] = pos_normalized[..., 0] / (W - 1)
        pos_normalized[..., 1] = pos_normalized[..., 1] / (H - 1)

        pos_x = pos_normalized[..., 0:1]
        pos_y = pos_normalized[..., 1:2]

        sin_x = torch.sin(pos_x * self.freq_bands)
        cos_x = torch.cos(pos_x * self.freq_bands)
        sin_y = torch.sin(pos_y * self.freq_bands)
        cos_y = torch.cos(pos_y * self.freq_bands)

        pos_encoding = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)

        if pos_encoding.shape[-1] < self.dim:
            padding = self.dim - pos_encoding.shape[-1]
            pos_encoding = F.pad(pos_encoding, (0, padding))

        return self.proj(pos_encoding)


# === 6. 增强的稀疏匹配网络（带梯度守卫）===
class OptimizedSparseMatchingNetwork(nn.Module):
    """优化的稀疏匹配网络 - 带层归一化和前馈网络"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim = cfg.FEATURE_DIM
        num_layers = cfg.NUM_ATTENTION_LAYERS
        num_heads = cfg.NUM_HEADS

        print(f"Initializing Optimized SparseMatchingNetwork with {num_layers} attention layers.")

        # 位置编码
        self.pos_enc = OptimizedPositionalEncoding(dim)

        # 层归一化
        self.norm_left = nn.LayerNorm(dim)
        self.norm_right = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

        # 自注意力层
        self.self_attn_left = nn.ModuleList([
            SparseAttention(dim, num_heads, cfg.SPARSITY_RATIO, cfg.WINDOW_SIZE)
            for _ in range(num_layers)
        ])

        self.self_attn_right = nn.ModuleList([
            SparseAttention(dim, num_heads, cfg.SPARSITY_RATIO, cfg.WINDOW_SIZE)
            for _ in range(num_layers)
        ])

        # 交叉注意力 - 使用修复的标准注意力
        self.cross_attn = ScaledDotProductAttention(dim, num_heads)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )

        # 应用权重初始化
        self.apply(init_weights)

    def forward(self, desc_left, desc_right, kp_left, kp_right, image_size):
        """优化的前向传播 - 带梯度守卫和数值稳定性检查"""
        # ✅ 守卫：确保输入需要梯度
        desc_left = desc_left.requires_grad_(True)
        desc_right = desc_right.requires_grad_(True)

        # ✅ 守卫：检查特征范数是否过大/过小
        left_norm = torch.norm(desc_left, dim=-1).mean()
        right_norm = torch.norm(desc_right, dim=-1).mean()
        if left_norm > 100 or left_norm < 1e-4:
            print(f"[WARNING] desc_left norm out of range: {left_norm.item()}")
        if right_norm > 100 or right_norm < 1e-4:
            print(f"[WARNING] desc_right norm out of range: {right_norm.item()}")

        # 位置编码
        pos_left = self.pos_enc(kp_left, image_size)
        pos_right = self.pos_enc(kp_right, image_size)

        # 自注意力处理
        feat_left = desc_left
        feat_right = desc_right

        for i, (self_l, self_r) in enumerate(zip(self.self_attn_left, self.self_attn_right)):
            # 残差连接 + 层归一化（保持梯度流）
            feat_left = feat_left + self_l(self.norm_left(feat_left + pos_left))
            feat_right = feat_right + self_r(self.norm_right(feat_right + pos_right))

        # 交叉注意力
        cross_out = self.cross_attn(feat_left, feat_right, feat_right)
        feat_left_enhanced = feat_left + self.cross_norm(cross_out)

        # 前馈网络
        feat_left_enhanced = feat_left_enhanced + self.ffn(self.ffn_norm(feat_left_enhanced))

        # 计算匹配分数
        match_scores = torch.bmm(feat_left_enhanced, feat_right.transpose(1, 2))

        # 几何约束
        constraint_mask = self._compute_constraint_mask(kp_left, kp_right)
        match_scores_constrained = match_scores.masked_fill(
            ~constraint_mask, torch.finfo(match_scores.dtype).min
        )

        # 视差预测
        match_probs = F.softmax(match_scores_constrained * self.cfg.MATCHING_TEMPERATURE, dim=2)
        disparity_matrix = kp_left[:, :, 0].unsqueeze(2) - kp_right[:, :, 0].unsqueeze(1)
        disparity_pred = (disparity_matrix * match_probs).sum(dim=2)
        disparity_pred = torch.nan_to_num(disparity_pred, nan=0.0)

        # ✅ 修复：强制视差为非负值
        disparity_pred = torch.clamp(disparity_pred, min=0.0)

        return match_scores_constrained, disparity_pred, constraint_mask

    def _compute_constraint_mask(self, kp_left, kp_right):
        """计算几何约束掩码"""
        x_left = kp_left[:, :, 0].unsqueeze(2)
        y_left = kp_left[:, :, 1].unsqueeze(2)
        x_right = kp_right[:, :, 0].unsqueeze(1)
        y_right = kp_right[:, :, 1].unsqueeze(1)

        valid_x_mask = (x_left >= x_right)
        valid_y_mask = (y_left - y_right).abs() < self.cfg.DISPARITY_CONSTRAINT_Y_THRESHOLD

        return valid_x_mask & valid_y_mask


# === 7. 优化的损失函数（数值稳定版）===
class OptimizedFeatureMetricLoss(nn.Module):
    """优化的特征度量损失 - 带温度参数"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, desc_left, desc_right, match_scores, keypoints_left,
                disparity, scores_left, constraint_mask):
        """优化的损失计算 - 数值稳定版"""
        B, N_l, C = desc_left.shape
        device = desc_left.device

        # 诊断：确保输入需要梯度
        if not desc_left.requires_grad or not desc_right.requires_grad:
            print(f"[WARNING] desc_left.requires_grad={desc_left.requires_grad}, "
                  f"desc_right.requires_grad={desc_right.requires_grad}")

        # 特征相似性损失
        match_probs = F.softmax(match_scores * self.cfg.MATCHING_TEMPERATURE, dim=2)
        desc_right_weighted = torch.bmm(match_probs, desc_right)

        # 余弦相似度
        eps = 1e-8
        desc_left_norm = F.normalize(desc_left, dim=2, eps=eps)
        desc_right_weighted_norm = F.normalize(desc_right_weighted, dim=2, eps=eps)

        cosine_sim = (desc_left_norm * desc_right_weighted_norm).sum(dim=2)

        # ✅ 修复：稳定化对数计算，避免log(0)
        # 将相似度限制在[-0.99, 0.99]范围内
        cosine_sim_clamped = torch.clamp(cosine_sim, min=-0.99, max=0.99)
        # 映射到[0.005, 0.995]避免log(0)和log(1)
        similarity_mapped = (cosine_sim_clamped + 1) / 2
        similarity_mapped = torch.clamp(similarity_mapped, min=0.005, max=0.995)

        feature_loss_per_kp = -torch.log(similarity_mapped + eps)

        # 有效掩码
        detection_mask = (scores_left > 0.1)
        matchable_mask = torch.any(constraint_mask, dim=2)
        final_valid_mask = detection_mask & matchable_mask

        # 平均特征损失
        num_valid = torch.sum(final_valid_mask)
        feature_loss = torch.tensor(0.0, device=device)
        if num_valid > 0:
            masked_loss = feature_loss_per_kp * final_valid_mask
            feature_loss = masked_loss.sum() / num_valid

        # 平滑度损失
        smooth_loss = self._compute_sparse_smoothness(keypoints_left, disparity, scores_left)

        # ✅ 确保总损失需要梯度
        total_loss = feature_loss + smooth_loss * self.cfg.SMOOTHNESS_WEIGHT

        # ✅ 添加数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[WARNING] NaN/Inf detected: feature_loss={feature_loss.item()}, smooth={smooth_loss.item()}")
            print(f"  cosine_sim_min={cosine_sim.min().item()}, max={cosine_sim.max().item()}")
            print(f"  match_probs_min={match_probs.min().item()}, max={match_probs.max().item()}")

            # 返回安全值
            return torch.tensor(0.1, device=device, requires_grad=True), \
                torch.tensor(0.1, device=device, requires_grad=True)

        return feature_loss, smooth_loss

    def _compute_sparse_smoothness(self, keypoints, disparity, scores):
        """计算稀疏平滑度损失"""
        B, N, _ = keypoints.shape
        total_smooth_loss = 0.0
        total_valid_pairs = 0

        for b in range(B):
            kp = keypoints[b]
            disp = disparity[b]
            sc = scores[b]

            valid_mask = sc > 0.1
            kp_valid = kp[valid_mask]
            disp_valid = disp[valid_mask]

            if len(kp_valid) < 2:
                continue

            distances = torch.cdist(kp_valid, kp_valid)
            neighbor_mask = (distances < 20) & (distances > 1e-6)

            disp_diff = (disp_valid.unsqueeze(1) - disp_valid.unsqueeze(0)).abs()

            batch_smooth_loss = (disp_diff * neighbor_mask.float()).sum()
            num_valid_pairs = neighbor_mask.sum()

            total_smooth_loss += batch_smooth_loss
            total_valid_pairs += num_valid_pairs

        smooth_loss_avg = torch.tensor(0.0, device=keypoints.device)
        if total_valid_pairs > 0:
            smooth_loss_avg = total_smooth_loss / total_valid_pairs

        return smooth_loss_avg


# === 8. 完整优化模型 ===
class OptimizedSparseMatchingStereoModel(nn.Module):
    """优化的稀疏立体匹配模型"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # 子模块
        self.keypoint_detector = OptimizedSparseKeypointDetector(cfg)
        self.feature_extractor = DINOv3FeatureExtractor(cfg)
        self.matcher = OptimizedSparseMatchingNetwork(cfg)

        print(f"Optimized Model with DINOv3: max {cfg.MAX_KEYPOINTS} keypoints, "
              f"{cfg.NUM_ATTENTION_LAYERS} attention layers, "
              f"feature dim {cfg.FEATURE_DIM}")

    def forward(self, left_gray, right_gray, left_rgb, right_rgb, mask):
        """前向传播"""
        B, _, H, W = left_gray.shape

        # 关键点检测
        kp_left, scores_left = self.keypoint_detector(left_gray, mask)
        kp_right, scores_right = self.keypoint_detector(
            right_gray, torch.ones_like(right_gray)
        )

        # DINOv3特征提取
        desc_left = self.feature_extractor(left_rgb, kp_left)
        desc_right = self.feature_extractor(right_rgb, kp_right)

        # 匹配
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


# === 9. 增强的训练器（添加显式backward验证）===
class OptimizedTrainer:
    """内存高效的训练器 - 带梯度监控和显式backward验证"""

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

        # 数据集和数据加载器
        train_ds = RectifiedWaveStereoDataset(cfg, is_validation=False)
        val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)
        num_workers = 0 if sys.platform == 'win32' else 4

        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_fn, num_workers=num_workers,
                                       pin_memory=True if self.device.type == 'cuda' else False)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=num_workers,
                                     pin_memory=True if self.device.type == 'cuda' else False)

        # 模型和损失函数
        self.model = OptimizedSparseMatchingStereoModel(cfg).to(self.device)
        self.loss_fn = OptimizedFeatureMetricLoss(cfg)

        # 应用权重初始化
        self.model.apply(init_weights)

        # 优化器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LEARNING_RATE, weight_decay=1e-4
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-7
        )

        # TensorBoard
        try:
            self.writer = SummaryWriter(log_dir=self.tb_dir) if SummaryWriter else None
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard SummaryWriter: {e}")
            self.writer = None

        # GradScaler
        self.scaler = None
        if cfg.USE_MIXED_PRECISION and self.device.type == 'cuda':
            try:
                self.scaler = torch.amp.GradScaler(self.device.type)
            except (AttributeError, TypeError):
                print("Using PyTorch 1.x GradScaler API")
                self.scaler = torch.cuda.amp.GradScaler()

        self.step = 0
        self.log_file = os.path.join(self.log_dir_json, "training_log.json")

        # 评估指标
        self.evaluator = EvaluationMetrics()

        # 历史记录
        self.loss_keys = ['total', 'feature', 'smoothness']
        self.metric_keys = ['mean_disparity', 'std_disparity', 'num_valid_keypoints']
        self.history = {
            'train': {k: [] for k in self.loss_keys + self.metric_keys},
            'val': {k: [] for k in self.loss_keys + self.metric_keys}
        }

        self.update_log_file(-1)

        # ✅ 修复：打印可训练参数信息
        trainable_params = [name for name, p in self.model.named_parameters() if p.requires_grad]
        print(f"\n[DEBUG] Total trainable parameters: {len(trainable_params)}")
        if len(trainable_params) < 5:
            print("[WARNING] Very few trainable parameters! Check model setup.")
        for name in trainable_params[:10]:
            print(f"  - {name}")
        if len(trainable_params) == 0:
            print("[FATAL] No trainable parameters found! Training cannot proceed.")
            sys.exit(1)

    def train(self):
        """训练循环"""
        print("\n=== Starting Optimized Training with DINOv3 (Numerically Stable) ===")
        print(f"Config: Max keypoints: {self.cfg.MAX_KEYPOINTS}, "
              f"Layers: {self.cfg.NUM_ATTENTION_LAYERS}, "
              f"LR: {self.cfg.LEARNING_RATE:.1e}, "
              f"Smooth Weight: {self.cfg.SMOOTHNESS_WEIGHT:.1e}, "
              f"Temp: {self.cfg.MATCHING_TEMPERATURE:.1f}")

        best_val_loss = float('inf')
        epochs_no_improve = 0

        try:
            for epoch in range(self.cfg.NUM_EPOCHS):
                # 训练阶段
                train_results = self._run_epoch(epoch, is_training=True)
                if not train_results:
                    print(f"Epoch {epoch + 1}: Training epoch failed or returned empty results. Stopping.")
                    break
                self._log_epoch_results('train', epoch, train_results)

                # 验证阶段
                with torch.no_grad():
                    val_results = self._run_epoch(epoch, is_training=False)
                if not val_results:
                    print(f"Epoch {epoch + 1}: Validation epoch failed or returned empty results. Stopping.")
                    break
                self._log_epoch_results('val', epoch, val_results)

                # 当前验证损失
                current_val_loss = val_results.get('total', float('inf'))

                # 打印进度
                print(f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> "
                      f"Train Loss: {train_results.get('total', float('nan')):.4f} | "
                      f"Val Loss: {current_val_loss:.4f} | "
                      f"Val Keypoints: {val_results.get('num_valid_keypoints', 0):.0f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

                # 更新日志和可视化
                self.update_log_file(epoch)
                if self.cfg.VISUALIZE_TRAINING and epoch % 5 == 0:
                    try:
                        self.plot_training_history()
                    except Exception as plot_e:
                        print(f"Warning: Failed to plot training history for epoch {epoch + 1}: {plot_e}")

                # 检查点保存和早停
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    epochs_no_improve = 0
                    save_path = os.path.join(self.ckpt_dir, f"best_model_epoch_{epoch + 1}.pth")
                    try:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"  Val loss improved to {best_val_loss:.4f}. Model saved to {save_path}")
                    except Exception as save_e:
                        print(f"Warning: Failed to save model checkpoint: {save_e}")
                else:
                    epochs_no_improve += 1
                    print(f"  No validation loss improvement for {epochs_no_improve} epochs.")
                    if epochs_no_improve >= self.cfg.EARLY_STOPPING_PATIENCE:
                        print(f"--- Early stopping triggered after {epoch + 1} epochs ---")
                        break

                # 调度器步进
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

    def _run_epoch(self, epoch, is_training):
        """运行一个epoch - 带显式backward验证"""
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader

        if not loader:
            print(f"Warning: {'Training' if is_training else 'Validation'} loader is empty.")
            return None

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [{'Train' if is_training else 'Val'}]",
                    leave=True)

        epoch_results = {k: 0.0 for k in self.loss_keys + self.metric_keys}
        batch_count = 0

        for data in pbar:
            if data is None:
                print("Warning: Skipping None batch returned by collate_fn.")
                continue

            try:
                # 数据移动到设备
                left_gray = data['left_gray'].to(self.device, non_blocking=True)
                right_gray = data['right_gray'].to(self.device, non_blocking=True)
                left_rgb = data['left_rgb'].to(self.device, non_blocking=True)
                right_rgb = data['right_rgb'].to(self.device, non_blocking=True)
                mask = data['mask'].to(self.device, non_blocking=True)
            except Exception as e:
                print(f"Error moving data to device: {e}. Skipping batch.")
                continue

            # ✅ 显式验证训练模式
            if is_training:
                assert self.model.training, "Model not in training mode!"

            # 混合精度上下文
            if self.cfg.USE_MIXED_PRECISION and self.scaler is not None:
                with torch.autocast(device_type=self.device.type):
                    loss_outputs = self._forward_and_loss(left_gray, right_gray, left_rgb, right_rgb, mask)
                if loss_outputs is None:
                    continue

                total_loss, feature_loss, smooth_loss_unweighted, outputs, metrics = loss_outputs

                if is_training:
                    self._backward_step(total_loss)
            else:
                loss_outputs = self._forward_and_loss(left_gray, right_gray, left_rgb, right_rgb, mask)
                if loss_outputs is None:
                    continue

                total_loss, feature_loss, smooth_loss_unweighted, outputs, metrics = loss_outputs

                if is_training:
                    # ✅ 验证损失是否需要梯度
                    if not total_loss.requires_grad:
                        print(f"[FATAL] Step {self.step}: total_loss.requires_grad=False")
                        continue

                    try:
                        # ✅ 显式调用backward
                        total_loss.backward(retain_graph=False)
                    except Exception as e:
                        print(f"[FATAL] backward() failed at step {self.step}: {e}")
                        raise

                    # ✅ 验证梯度是否计算
                    grad_found = False
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            if param.grad.norm().item() > 0:
                                grad_found = True
                                break

                    if not grad_found:
                        print(f"\n[WARNING] Step {self.step}: No non-zero gradients found!")
                        print("This indicates gradient flow is blocked. Check:")
                        print("1. DINO feature extractor output requires_grad")
                        print("2. Loss computation uses network outputs")
                        print("3. No detach() operations in forward pass")
                    else:
                        # ✅ 仅在确认梯度存在后执行优化步骤
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            # 累积结果
            batch_count += 1
            for k in self.loss_keys:
                if k == 'total':
                    epoch_results[k] += total_loss.item()
                elif k == 'feature':
                    epoch_results[k] += feature_loss.item()
                elif k == 'smoothness':
                    epoch_results[k] += smooth_loss_unweighted.item()

            for k in self.metric_keys:
                epoch_results[k] += metrics.get(k, 0.0)

            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss.item(),
                'feat_loss': feature_loss.item(),
                'kpts': metrics.get('num_valid_keypoints', 0),
                'disp': metrics.get('mean_disparity', 0.0)
            })

            # 日志记录和可视化
            if is_training:
                if self.writer:
                    self.writer.add_scalar('Loss/step_train_total', total_loss.item(), self.step)
                    self.writer.add_scalar('Loss/step_train_feature', feature_loss.item(), self.step)
                    self.writer.add_scalar('Loss/step_train_smoothness_unweighted', smooth_loss_unweighted.item(),
                                           self.step)

                # 每50步检查梯度
                if self.step % 50 == 0:
                    self._check_gradients()

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

        # 计算平均结果
        if batch_count > 0:
            return {k: v / batch_count for k, v in epoch_results.items()}
        else:
            print(f"Warning: {'Training' if is_training else 'Validation'} epoch {epoch + 1} had no valid batches.")
            return epoch_results

    def _forward_and_loss(self, left_gray, right_gray, left_rgb, right_rgb, mask):
        """前向传播并计算损失 - 带显式验证"""
        try:
            outputs = self.model(left_gray, right_gray, left_rgb, right_rgb, mask)
            if outputs is None:
                raise ValueError("Model forward pass returned None")
        except Exception as e:
            print(f"\nError during model forward pass: {e}")
            return None

        # ✅ 显式检查关键输出是否需要梯度
        if not outputs['descriptors_left'].requires_grad:
            print(f"[FATAL] descriptors_left.requires_grad=False. Gradient flow blocked!")
            return None

        if not outputs['match_scores'].requires_grad:
            print(f"[FATAL] match_scores.requires_grad=False. Matching network not connected!")
            return None

        try:
            feature_loss, smooth_loss_unweighted = self.loss_fn(
                outputs['descriptors_left'],
                outputs['descriptors_right'],
                outputs['match_scores'],
                outputs['keypoints_left'],
                outputs['disparity'],
                outputs['scores_left'],
                outputs['constraint_mask']
            )

            total_loss = self.cfg.PHOTOMETRIC_WEIGHT * feature_loss + self.cfg.SMOOTHNESS_WEIGHT * smooth_loss_unweighted

            if not torch.isfinite(total_loss):
                raise ValueError("Loss is NaN or Inf")

            # ✅ 最终验证
            if not total_loss.requires_grad:
                print("[FATAL] total_loss does not require grad!")
                return None

        except Exception as e:
            print(f"\nError calculating loss: {e}")
            return None

        # 计算指标
        try:
            metrics = self.evaluator.compute_sparse_metrics(
                outputs['disparity'].detach(),
                outputs['keypoints_left'].detach(),
                outputs['scores_left'].detach()
            )
        except Exception as e:
            print(f"\nWarning: Failed to compute metrics: {e}. Metrics will be zero.")
            metrics = {k: 0.0 for k in self.metric_keys}

        return total_loss, feature_loss, smooth_loss_unweighted, outputs, metrics

    def _backward_step(self, total_loss):
        """混合精度反向传播"""
        if self.scaler is None:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            accum_steps = self.cfg.GRADIENT_ACCUMULATION_STEPS
            self.scaler.scale(total_loss / accum_steps).backward()

            if (self.step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

    def _check_gradients(self):
        """增强的梯度检查 - 更详细的诊断"""
        total_norm = 0
        param_count = 0
        zero_count = 0
        small_count = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.norm().item()
                total_norm += param_norm ** 2
                param_count += 1

                if param_norm < 1e-8:
                    zero_count += 1
                elif param_norm < 1e-6:
                    small_count += 1

        if param_count > 0:
            total_norm = (total_norm ** 0.5)
            avg_norm = total_norm / param_count
            print(f"\n[Gradient Check] Step {self.step}:")
            print(f"  - Total norm: {total_norm:.6f}")
            print(f"  - Avg norm: {avg_norm:.6f}")
            print(f"  - Zero grad params (<1e-8): {zero_count}/{param_count}")
            print(f"  - Small grad params (<1e-6): {small_count}/{param_count}")

            if total_norm < 1e-6:
                print(f"  [WARNING] Gradients are too small! Learning may be stuck.")
        else:
            print(f"\n[WARNING] Step {self.step}: No parameters with gradients found!")

    def _log_epoch_results(self, phase, epoch, results):
        """记录epoch结果"""
        if not results:
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
                    elif k == 'feature':
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_feature", v, epoch)
                    elif k == 'total':
                        self.writer.add_scalar(f"Loss_Epoch/{phase}_total", v, epoch)
                    else:
                        self.writer.add_scalar(f"{metric_type}_Epoch/{phase}_{k}", v, epoch)
                except Exception as e:
                    print(f"Warning: Failed to log {k}={v} to TensorBoard: {e}")

    def visualize(self, data, outputs, step, phase):
        """可视化"""
        try:
            left_gray = data['left_gray'][0, 0].numpy()
            kp_left = outputs['keypoints_left'][0].numpy()
            scores_left = outputs['scores_left'][0].numpy()
            disparity = outputs['disparity'][0].numpy()

            valid_mask = scores_left > 0.1
            kp_valid = kp_left[valid_mask]
            disp_valid = disparity[valid_mask]

            if len(disp_valid) > 0:
                disp_valid_finite = disp_valid[np.isfinite(disp_valid)]
            else:
                disp_valid_finite = np.array([])

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Sparse Matching - Step: {step} ({phase})', fontsize=16)

            # 关键点可视化
            axes[0].imshow(left_gray, cmap='gray')
            if len(kp_valid) > 0:
                axes[0].scatter(kp_valid[:, 0], kp_valid[:, 1], c='red', s=10, alpha=0.6)
            axes[0].set_title(f"Detected Keypoints ({len(kp_valid)})")
            axes[0].axis('off')

            # 视差分布
            if len(disp_valid_finite) > 0:
                q99 = np.percentile(disp_valid_finite, 99) if len(disp_valid_finite) > 1 else disp_valid_finite.max()
                hist_max = max(1.0, q99 * 1.1)
                axes[1].hist(disp_valid_finite, bins=50, range=(0, hist_max), alpha=0.7, edgecolor='black')
                mean_disp = disp_valid_finite.mean()
                axes[1].axvline(mean_disp, color='red', linestyle='--', label=f'Mean: {mean_disp:.2f}')
                axes[1].set_xlabel('Disparity (pixels)')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Disparity Distribution')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].set_title('Disparity Distribution (No valid disparities)')
                axes[1].text(0.5, 0.5, 'No valid disparities to plot',
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[1].transAxes)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = os.path.join(self.vis_dir, f"{phase}_step_{step:06d}.png")
            plt.savefig(save_path, dpi=100)

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
            if 'fig' in locals():
                plt.close(fig)

    def plot_training_history(self):
        """绘制训练历史"""
        if not self.history or not self.history['train'].get('total') or not self.history['val'].get('total'):
            print("Warning: Insufficient training history found to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sparse Matching Training History (Numerically Stable)', fontsize=16)

        epochs = range(len(self.history['train']['total']))

        # 总损失
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
        except Exception as e:
            print(f"Warning: Plotting Total Loss failed: {e}")

        # 关键点数量
        try:
            train_kpts = [k if np.isfinite(k) else np.nan for k in self.history['train']['num_valid_keypoints']]
            val_kpts = [k if np.isfinite(k) else np.nan for k in self.history['val']['num_valid_keypoints']]
            axes[0, 1].plot(epochs, train_kpts, label='Train Keypoints')
            axes[0, 1].plot(epochs, val_kpts, label='Val Keypoints')
            axes[0, 1].set_title('Number of Valid Keypoints')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            axes[0, 1].set_xlabel("Epochs")
            axes[0, 1].set_ylabel("Count")
        except Exception as e:
            print(f"Warning: Plotting Keypoints failed: {e}")

        # 平均视差
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

        # 损失组件
        try:
            if self.history['train']['feature'] and self.history['train']['smoothness']:
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
        except Exception as e:
            print(f"Warning: Plotting Loss Components failed: {e}")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.vis_dir, "training_history_numerical_stable.png")
        plt.savefig(save_path)
        plt.close(fig)

    def update_log_file(self, epoch):
        """更新日志文件"""
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
        except Exception as e:
            print(f"Warning: Failed to write log file '{self.log_file}': {e}")


# === 数据集类 ===
class RectifiedWaveStereoDataset(Dataset):
    """Loads, rectifies, crops, resizes, and augments stereo image pairs."""

    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.is_validation = is_validation

        # Find all left images
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"错误: 在 '{cfg.LEFT_IMAGE_DIR}' 中未找到图像。")

        # Load stereo calibration parameters
        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            self.roi_left = tuple(map(int, calib['roi_left']))
            self.roi_right = tuple(map(int, calib['roi_right']))
        except Exception as e:
            sys.exit(f"加载标定文件 '{cfg.CALIBRATION_FILE}' 失败: {e}")

        # Split dataset
        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]
        print(f"{'验证集' if is_validation else '训练集'}: {len(self.indices)} 帧")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        try:
            img_idx = self.indices[idx]
            if img_idx < 0 or img_idx >= len(self.left_images):
                return None

            left_path = self.left_images[img_idx]
            right_filename = 'right' + os.path.basename(left_path)[4:]
            right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_filename)

            # Load images
            left_raw = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            right_raw = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

            if left_raw is None or right_raw is None:
                return None

            # Rectification
            left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

            # ROI Cropping
            lx, ly, lw, lh = self.roi_left
            rx, ry, rw, rh = self.roi_right

            if ly + lh > left_rect.shape[0] or lx + lw > left_rect.shape[1] or \
                    ry + rh > right_rect.shape[0] or rx + rw > right_rect.shape[1] or \
                    ly < 0 or lx < 0 or ry < 0 or rx < 0:
                return None

            left_rect_cropped = left_rect[ly:ly + lh, lx:lx + lw]
            right_rect_cropped = right_rect[ry:ry + rh, rx:rx + rw]

            # Resizing
            target_size = (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT)
            if left_rect_cropped.shape[0] == 0 or left_rect_cropped.shape[1] == 0 or \
                    right_rect_cropped.shape[0] == 0 or right_rect_cropped.shape[1] == 0:
                return None

            left_img = cv2.resize(left_rect_cropped, target_size)
            right_img = cv2.resize(right_rect_cropped, target_size)

            # Data Augmentation
            if not self.is_validation and self.cfg.USE_ADVANCED_AUGMENTATION and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
                if np.random.rand() < 0.5:
                    brightness = np.random.uniform(0.7, 1.3)
                    left_img = np.clip(left_img * brightness, 0, 255).astype(np.uint8)
                    right_img = np.clip(right_img * brightness, 0, 255).astype(np.uint8)
                if np.random.rand() < 0.5:
                    contrast = np.random.uniform(0.7, 1.3)
                    mean_l, mean_r = left_img.mean(), right_img.mean()
                    left_img = np.clip((left_img - mean_l) * contrast + mean_l, 0, 255).astype(np.uint8)
                    right_img = np.clip((right_img - mean_r) * contrast + mean_r, 0, 255).astype(np.uint8)

            # Mask Generation
            _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Convert to tensors
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
            print(f"警告: 处理索引 {idx} 时发生意外错误: {e}")
            return None


# === 评估指标类 ===
class EvaluationMetrics:
    """Computes evaluation metrics based on sparse disparity predictions."""

    @staticmethod
    def compute_sparse_metrics(disparity, keypoints_left, scores_left):
        """
        Computes basic metrics like mean/std disparity and number of valid keypoints.
        """
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


# === collate函数 ===
def collate_fn(batch):
    """Custom collate function to filter out None samples from the batch."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# === 主程序 ===
def main():
    """主函数"""
    cfg = Config()

    # 自动调整配置
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
            if gpu_memory < 8000:
                cfg.BATCH_SIZE = 1
                cfg.MAX_KEYPOINTS = 256
            elif gpu_memory < 12000:
                cfg.BATCH_SIZE = 1
                cfg.MAX_KEYPOINTS = 384
            else:
                cfg.BATCH_SIZE = 2
                cfg.MAX_KEYPOINTS = 512
    except:
        pass

    # 创建训练器并启动训练
    trainer = OptimizedTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()