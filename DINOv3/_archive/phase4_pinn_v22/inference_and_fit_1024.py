import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
import glob  # 导入 glob 用于查找文件

# ----------------------------------------------------------------------
# --- [ V17 最终修复 ] ---
# 诊断: V14-V16 日志显示，无论是 V14 (注意力匹配) 还是 V15/V16 (暴力匹配)，
#       都产生了过多的噪声点 (局外点)，导致拟合失败。
#       V14 的问题是它没有使用匹配器的“置信度”来过滤匹配。
# 策略:
# 1. 恢复 V14 的策略：使用 V9 训练的完整模型 (Detector + Extractor + Matcher)。
# 2. 保持 V13 的坐标系/视差修复 (u, v, d)。
# 3. [V17 关键修复] 在 main() 函数中，我们现在将正确地
#    从 'match_scores' 中提取匹配置信度 (最大概率)，
#    并只保留置信度 > 0.2 的匹配。
# ----------------------------------------------------------------------

try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80 + "\n[FATAL ERROR]: transformers not found. 请安装: pip install transformers\n" + "=" * 80)
    sys.exit(1)

# --- 路径定义 ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.dirname(PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
    DATA_ROOT = r"D:\Research\wave_reconstruction_project"

if not os.path.isdir(DATA_ROOT):
    print(f"[警告] 自动检测到的 DATA_ROOT 路径不存在: {DATA_ROOT}")
    print(f"       将使用硬编码路径 D:\\Research\\wave_reconstruction_project")
    PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
    DATA_ROOT = r"D:\Research\wave_reconstruction_project"


# --- 路径定义结束 ---


# --- 1. 配置 (Config) ---
@dataclass
class Config:
    """
    配置 - 必须与 v9 训练脚本 (sparse_reconstructor_1115_gemini.py)
    中的 Config *完全一致*。
    """
    # 来自 1024 脚本 (模型架构)
    NUM_ATTENTION_LAYERS: int = 6
    FEATURE_DIM: int = 768
    NUM_HEADS: int = 8
    DISPARITY_CONSTRAINT_Y_THRESHOLD: int = 2  # V9 训练时使用的是 2
    MATCHING_TEMPERATURE: float = 10.0

    # DINO 路径
    DINO_LOCAL_PATH: str = "dinov3-base-model"

    # --- [ v9 关键修改 ] ---
    # 必须与 v9 训练脚本的参数 *完全一致*
    MAX_KEYPOINTS: int = 512
    MASK_THRESHOLD: int = 30

    BLOB_MIN_THRESHOLD: float = 30.0

    # 1. 面积 (Area)
    BLOB_MIN_AREA: float = 20.0  # (捕获远处的点)
    BLOB_MAX_AREA: float = 2000.0  # (应对近处的亮团)

    # 2. 圆度 (Circularity)
    BLOB_MIN_CIRCULARITY: float = 0.1

    # 3. 凸性 (Convexity)
    BLOB_MIN_CONVEXITY: float = 0.90

    # 4. 惯性比 (Inertia Ratio)
    BLOB_MIN_INERTIA: float = 0.1
    BLOB_MAX_INERTIA: float = 0.85  # (允许 "亮白团" 通过)
    # --- [ v9 修改结束 ] ---


# --- 2. [ 关键替换 ] ---
# 我们使用来自 'sparse_reconstructor_1030_gemini.py' 的 Blob 检测器
class SparseKeypointDetector(nn.Module):
    """
    使用 OpenCV 的 SimpleBlobDetector 检测稀疏关键点。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_keypoints = cfg.MAX_KEYPOINTS

        # --- 设置 Blob 检测器 (使用 v9 Config) ---
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
        params.maxInertiaRatio = cfg.BLOB_MAX_INERTIA  # <-- [v9] 使用新上限

        self.detector = cv2.SimpleBlobDetector_create(params)
        print(f"--- [检测器] 已初始化 SimpleBlobDetector (v17 推理模式) ---")
        print(f"    阈值: [{params.minThreshold}, {params.maxThreshold}]")
        print(f"    面积 (Area): [{params.minArea}, {params.maxArea}]")
        print(f"    凸性 (Convexity): > {params.minConvexity}")
        print(f"    惯性 (Inertia): [{params.minInertiaRatio}, {params.maxInertiaRatio}]")
        print(f"-------------------------------------------------------")

    def forward(self, gray_image, mask):
        """
        Args:
            gray_image: (B, 1, H, W) - 灰度图像 (0-1 范围)
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

            # [ v8/v9 BUG 修复 ]
            # 直接在原始灰度图 (img_tensor) 上操作
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)

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


# --- 3. DINOv3 特征提取器 (DINOv3FeatureExtractor) ---
class DINOv3FeatureExtractor(nn.Module):
    """从 1024 版本复制: 提取 DINO 特征"""

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
            model_path = self.cfg.DINO_LOCAL_PATH
            if os.path.isdir(model_path):
                print(f"[信息] 从本地路径加载 DINO: {model_path}")
                return AutoModel.from_pretrained(model_path, local_files_only=True)
            else:
                print(f"[警告] DINO 本地路径 {model_path} 未找到。")
                print(f"       尝试从 Hugging Face Hub 加载: {model_path}")
                if 'v3' in model_path:
                    print("[严重警告] 你的路径包含 'v3'。DINOv3 并非公开模型。")
                    print("       如果你想加载 DINOv2, 请使用 'facebook/dinov2-base'。")
                return AutoModel.from_pretrained(model_path)
        except Exception as e:
            print(f"[致命] 加载 DINO 模型失败: {e}")
            sys.exit(1)

    def get_feature_map(self, image):
        with torch.no_grad():
            outputs = self.dino(image)
            features = outputs.last_hidden_state
        b, _, h, w = image.shape
        start_idx = 1 + self.num_register_tokens
        if features.shape[1] <= start_idx:
            raise RuntimeError(f"DINO token 数量不足 ({features.shape[1]})，无法跳过 CLS/REG tokens (需要 {start_idx})。")
        patch_tokens = features[:, start_idx:, :]
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        expected_tokens = h_feat * w_feat
        if patch_tokens.shape[1] != expected_tokens:
            print(f"[警告] DINO patch token 数量 ({patch_tokens.shape[1]}) 与预期 ({expected_tokens}) 不符。")
            if patch_tokens.shape[1] > expected_tokens:
                print(f"  > 截断 token: {patch_tokens.shape[1]} -> {expected_tokens}")
                patch_tokens = patch_tokens[:, :expected_tokens, :]
            else:
                raise RuntimeError(
                    f"DINO token 数量不足 ({patch_tokens.shape[1]} < {expected_tokens})，请检查图像尺寸是否为 patch_size 的整数倍。")
        feat_map = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, h_feat, w_feat)
        return feat_map

    def forward(self, image, keypoints):
        B, N, _ = keypoints.shape
        feat_map = self.get_feature_map(image)
        _, C, H_feat, W_feat = feat_map.shape
        _, _, H_img, W_img = image.shape
        if W_img <= 1 or H_img <= 1:
            return torch.zeros((B, N, C), device=image.device)
        grid = keypoints.clone()
        grid[..., 0] = 2 * (grid[..., 0] / (W_img - 1)) - 1
        grid[..., 1] = 2 * (grid[..., 1] / (H_img - 1)) - 1
        grid = grid.unsqueeze(2)
        descriptors_sampled = F.grid_sample(
            feat_map, grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )
        descriptors = descriptors_sampled.squeeze(3).permute(0, 2, 1)
        return descriptors


# --- 4. [V14/V17 恢复] 注意力匹配网络 (从 V9 训练脚本复制) ---
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

        # [V17] 同时返回 match_probs 用于置信度过滤
        return match_scores_constrained, disparity_pred, constraint_mask, match_probs


# --- [V17 恢复结束] ---


class SparseMatchingStereoModel(nn.Module):
    """
    [ V17 修改 ]
    这个类现在是 V9 训练脚本的完整副本。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.keypoint_detector = SparseKeypointDetector(cfg)
        self.feature_extractor = DINOv3FeatureExtractor(cfg)
        self.matcher = SparseMatchingNetwork(cfg)  # <-- [V14/V17] 恢复

    def forward(self, left_gray, right_gray, left_rgb, right_rgb, mask):
        B, _, H, W = left_gray.shape

        # 1. 在左图上运行检测器
        kp_left, scores_left = self.keypoint_detector(left_gray, mask)

        # 2. 在右图上运行检测器
        kp_right, scores_right = self.keypoint_detector(right_gray, torch.ones_like(right_gray))

        # 3. 提取左图特征
        desc_left = self.feature_extractor(left_rgb, kp_left)

        # 4. 提取右图特征
        desc_right = self.feature_extractor(right_rgb, kp_right)

        # 5. [ V14/V17 ] 运行注意力匹配器
        match_scores, disparity, constraint_mask, match_probs = self.matcher(
            desc_left, desc_right, kp_left, kp_right, (H, W)
        )

        return {
            'keypoints_left': kp_left,
            'scores_left': scores_left,
            'descriptors_left': desc_left,
            'keypoints_right': kp_right,
            'scores_right': scores_right,
            'descriptors_right': desc_right,
            'match_scores': match_scores,
            'disparity': disparity,
            'constraint_mask': constraint_mask,
            'match_probs': match_probs  # [V17] 传出置信度
        }


# --- 5. (内部) 填充函数 ---
def _pad_inputs(patch_size, *tensors):
    """ 填充输入张量，使其 H 和 W 可被 patch_size 整除 """
    if patch_size is None: patch_size = 16

    if not tensors: return []
    if tensors[0] is None:
        raise ValueError("传入 _pad_inputs 的第一个张量为 None")
    b, c, h, w = tensors[0].shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padded_tensors = []
    if pad_h > 0 or pad_w > 0:
        for x in tensors:
            if x is not None:
                padded_tensors.append(F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0))
            else:
                padded_tensors.append(None)
        return padded_tensors
    return list(tensors)


# ----------------------------------------------------------------------
# 步骤 2: 重建和拟合函数 (已修改)
# ----------------------------------------------------------------------

def load_calibration(calib_file):
    """加载标定文件并返回 Q 矩阵和标定参数。"""
    try:
        calib = np.load(calib_file)
        if 'Q' not in calib:
            print(f"[错误] 标定文件 {calib_file} 中未找到 'Q' 矩阵。")
            sys.exit(1)
        map1_left, map2_left = calib['map1_left'], calib['map2_left']
        map1_right, map2_right = calib['map1_right'], calib['map2_right']
        roi_left, roi_right = tuple(map(int, calib['roi_left'])), tuple(map(int, calib['roi_right']))
        return calib['Q'], map1_left, map2_left, map1_right, map2_right, roi_left, roi_right
    except Exception as e:
        print(f"[错误] 无法加载标定文件 {calib_file}: {e}")
        sys.exit(1)


# --- [ V15 移除 ] ---
# def match_features_bruteforce(...)
# --- [ V15 移除结束 ] ---

def reproject_to_3d(keypoints_rect_space, disparity_rect_space, Q):  # [V13] 重命名
    """
    使用 Q 矩阵将稀疏点 (u, v, d) 重投影到 (X, Y, Z)。
    [V13] 假设: keypoints_rect_space 和 disparity_rect_space 都是
          *矫正后* 且 *加上了 ROI 偏移* 的坐标/视差
    """
    N = keypoints_rect_space.shape[0]
    if N == 0:
        return np.array([]).reshape(0, 3)
    kp = keypoints_rect_space
    disp = disparity_rect_space.reshape(-1, 1)
    ones = np.ones((N, 1))
    points_4d = np.hstack((kp, disp, ones))
    try:
        projected_points_4d = (Q @ points_4d.T).T
    except Exception as e:
        print(f"[错误] 3D 重投影矩阵乘法失败: {e}")
        return np.array([]).reshape(0, 3)

    W = projected_points_4d[:, 3].reshape(-1, 1)

    # [V11 关键修复] 检查 W 是否接近于 0
    W_safe = W.copy()
    zero_w_mask = np.abs(W) < 1e-9
    if np.any(zero_w_mask):
        print(f"[警告] {np.sum(zero_w_mask)} 个点的 W 坐标接近 0，可能导致 3D 坐标爆炸。")
        W_safe[zero_w_mask] = 1e-9  # 避免除以 0

    points_3d = projected_points_4d[:, :3] / W_safe

    # [V11] 也过滤掉那些 W 接近 0 的点
    points_3d = points_3d[~zero_w_mask.flatten()]

    if points_3d.shape[0] == 0:
        print("[警告] 3D 重投影后没有剩余的点 (可能所有 W 都为 0)。")
        return np.array([]).reshape(0, 3)

    return points_3d


def filter_point_cloud(points_3d, height_axis, prop_axis, z_max=15000, height_range=(-1000, 1000)):
    """
    过滤 3D 点云以移除无效值和物理上的不可能值。
    """
    if points_3d.shape[0] == 0:
        print(f"过滤 3D 点云: 原始 {points_3d.shape[0]} -> 过滤后 0")
        return points_3d

    if height_axis not in [0, 1, 2] or prop_axis not in [0, 1, 2] or height_axis == prop_axis:
        print(f"[错误] 无效的轴索引: height={height_axis}, propagation={prop_axis} (不能相同)")
        return np.array([])

    points_3d_corrected = points_3d

    print("--- 3D 点云原始范围 (过滤前) ---")
    finite_points = points_3d_corrected[np.isfinite(points_3d_corrected).all(axis=1)]
    if finite_points.shape[0] > 0:
        print(f"  X 轴 (索引 0) 范围: [{np.min(finite_points[:, 0]):.2f}, {np.max(finite_points[:, 0]):.2f}]")
        print(f"  Y 轴 (索引 1) 范围: [{np.min(finite_points[:, 1]):.2f}, {np.max(finite_points[:, 1]):.2f}]")
        print(f"  Z 轴 (索引 2) 范围: [{np.min(finite_points[:, 2]):.2f}, {np.max(finite_points[:, 2]):.2f}]")
        print(f"  (使用 Z < {z_max} 和 波高 ({height_axis}) 在 {height_range} 范围内进行过滤)")
    else:
        print("  [警告] 3D 点云不包含任何有效 (finite) 数据。")
    print("-----------------------------------")

    valid_mask_z = (points_3d_corrected[:, 2] < z_max) & (points_3d_corrected[:, 2] > 0)
    height_data = points_3d_corrected[:, height_axis]
    valid_mask_height = (height_data > height_range[0]) & (height_data < height_range[1])

    # [V11] 添加 finite 检查
    valid_mask_finite = np.isfinite(points_3d_corrected).all(axis=1)

    final_mask = valid_mask_z & valid_mask_height & valid_mask_finite

    if np.sum(final_mask) == 0 and len(points_3d_corrected) > 0:
        print("[调试] 所有点都被过滤了。原因分析:")
        print(f"  - {np.sum(~valid_mask_z)} / {len(points_3d_corrected)} 个点因 Z 轴范围 (0 < Z < {z_max}) 过滤失败")
        print(
            f"  - {np.sum(~valid_mask_height)} / {len(points_3d_corrected)} 个点因 波高 ({height_axis}轴) 范围 ({height_range}) 过滤失败")
        print(
            f"  - {np.sum(~valid_mask_finite)} / {len(points_3d_corrected)} 个点因 非法坐标 (inf/nan) 过滤失败")

    filtered_points = points_3d_corrected[final_mask]
    print(f"过滤 3D 点云: 原始 {len(points_3d_corrected)} -> 过滤后 {len(filtered_points)}")
    return filtered_points


def wave_model_func(prop_data, A, k, phi, y_offset):
    """定义波形函数。"""
    return A * np.cos(k * prop_data + phi) + y_offset


def fit_wave_parameters(points_3d, height_axis, prop_axis):
    """
    两阶段鲁棒拟合
    """
    min_points_for_fit = 10
    if points_3d.shape[0] < min_points_for_fit:
        print(f"[警告] 过滤后的点太少 (<{min_points_for_fit})，无法进行拟合。")
        return None, None, None

    prop_data = points_3d[:, prop_axis]
    height_data = points_3d[:, height_axis]

    # --- 阶段 1: 粗略拟合 ---
    print(f"开始 阶段 1 粗略拟合... (传播轴: {prop_axis}, 波高轴: {height_axis})")
    if np.max(height_data) == np.min(height_data):
        print("[警告] 所有点的高度都相同，无法猜测波幅。")
        A_guess = 1.0
    else:
        A_guess = (np.max(height_data) - np.min(height_data)) / 2
    y_offset_guess = np.mean(height_data)

    # [V11] 改进 k_guess, 基于论文
    # 严师兄的论文 (图 4-8) 显示波长 L ~ 2500mm
    # 刘师兄的论文 (第 4.2.3 节) 也显示波长 L ~ 2500mm
    L_guess = 2500
    k_guess = (2 * np.pi) / L_guess

    phi_guess = 0
    p0 = [A_guess, k_guess, phi_guess, y_offset_guess]
    print(f"初始猜测 (A, k, phi, y_offset): {p0[0]:.2f}, {p0[1]:.4f}, {p0[2]}, {p0[3]:.2f}")

    try:
        params_rough, _ = curve_fit(wave_model_func, prop_data, height_data, p0=p0, maxfev=5000)
    except RuntimeError as e:
        print(f"[错误] 阶段 1 粗略拟合失败: {e}")
        return None, None, None
    except Exception as e:
        print(f"[未知错误] 阶段 1 拟合中发生未知错误: {e}")
        return None, None, None

    # --- 阶段 2: 数据清洗 ---
    height_pred_rough = wave_model_func(prop_data, *params_rough)
    residuals = height_data - height_pred_rough
    res_std = np.std(residuals)
    median_abs_deviation = np.median(np.abs(residuals - np.median(residuals)))
    if median_abs_deviation == 0:
        median_abs_deviation = 1.0

    # [V11] 使用更严格的阈值
    inlier_threshold = 2.0 * 1.4826 * median_abs_deviation  # (原为 2.5)
    inlier_threshold = min(inlier_threshold, 100.0)  # 物理上限 100mm
    print(
        f"阶段 2 数据清洗: 残差 STD ~ {res_std:.2f}, MAD ~ {median_abs_deviation:.2f}, 使用阈值: {inlier_threshold:.2f} mm")
    inlier_mask = np.abs(residuals) < inlier_threshold
    num_inliers = np.sum(inlier_mask)

    if num_inliers < min_points_for_fit:
        print(f"[错误] 阶段 2: 局内点数量 ({num_inliers}) 太少 ( < {min_points_for_fit} )。")
        print("       这说明模型匹配噪声过大，或者初始猜测完全错误。")
        return None, None, None
    print(f"阶段 2 数据清洗: 保留 {num_inliers} / {len(prop_data)} 个局内点。")
    prop_inliers = prop_data[inlier_mask]
    height_inliers = height_data[inlier_mask]

    # --- 阶段 3: 精确拟合 ---
    print("开始 阶段 3 精确拟合...")
    try:
        params_final, covariance_final = curve_fit(wave_model_func, prop_inliers, height_inliers, p0=params_rough,
                                                   maxfev=5000)
        if params_final[0] < 0:
            params_final[0] = -params_final[0]
            params_final[2] += np.pi
        print("精确拟合成功。")
        return params_final, covariance_final, inlier_mask
    except RuntimeError as e:
        print(f"[错误] 阶段 3 精确拟合失败: {e}")
        return None, None, None
    except Exception as e:
        print(f"[未知错误] 阶段 3 拟合中发生未知错误: {e}")
        return None, None, None


def plot_reconstruction(points_3d, fit_params, inlier_mask, output_image, height_axis, prop_axis, common_dims):
    """
    可视化重建结果，区分局内点和局外点
    [v9 修复] 增加 common_dims 用于计算画布大小
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"[警告] 设置中文字体 'SimHei' 失败: {e}。 图像标签可能显示为方块。")

    axis_labels = {0: 'X 轴', 1: 'Y 轴', 2: 'Z 轴'}
    prop_label = axis_labels.get(prop_axis, f'轴 {prop_axis}')
    height_label = axis_labels.get(height_axis, f'轴 {height_axis}')
    prop_data = points_3d[:, prop_axis]
    height_data = points_3d[:, height_axis]

    # --- [ v9 修复: 清晰的预览图 ] ---
    # 根据图像的宽高比动态计算 figsize
    img_w, img_h = common_dims
    aspect_ratio = img_w / img_h
    # (这里我们只显示一个图，不是两个)
    fig_h = 10
    fig_w = fig_h * aspect_ratio
    if fig_w < 15: fig_w = 15  # 保持最小宽度

    plt.figure(figsize=(fig_w, fig_h))  # 使用动态计算的 figsize
    # --- [ v9 修复结束 ] ---

    if inlier_mask is not None:
        outlier_mask = ~inlier_mask
        plt.scatter(prop_data[outlier_mask], height_data[outlier_mask],
                    s=5, alpha=0.3, c='gray', label=f"局外点 (n={np.sum(outlier_mask)})")
        plt.scatter(prop_data[inlier_mask], height_data[inlier_mask],
                    s=10, alpha=0.8, c='blue', label=f"局内点 (n={np.sum(inlier_mask)})")
    else:
        plt.scatter(prop_data, height_data, s=5, alpha=0.6, label="稀疏 3D 点云")

    if fit_params is not None:
        A, k, phi, y_offset = fit_params
        prop_smooth = np.linspace(np.min(prop_data), np.max(prop_data), 500)
        height_fit = wave_model_func(prop_smooth, A, k, phi, y_offset)
        plt.plot(prop_smooth, height_fit, 'r-', linewidth=2, label="鲁棒拟合波形曲线")
        Amplitude = A * 2
        Wavelength = (2 * np.pi) / (k + 1e-9)
        title_text = f"参数化波形重建 (V17: V9 训练模型 + V13 坐标修复 + 置信度过滤)\n波高 (2*A): {Amplitude:.2f} mm | 波长 (L): {Wavelength:.2f} mm | 波数 (k): {k:.4f}"
        plt.title(title_text)
    else:
        plt.title("稀疏点云 (拟合失败)")

    plt.xlabel(f"传播方向 ({prop_label}) (mm)")
    plt.ylabel(f"波高 ({height_label}) (mm)")
    plt.legend()
    plt.grid(True)
    if height_axis == 1:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    try:
        # --- [ v9 修复: 清晰的预览图 ] ---
        plt.savefig(output_image, dpi=150)  # 使用更高的 DPI
        # --- [ v9 修复结束 ] ---
        print(f"重建可视化图像已保存到: {output_image}")
    except Exception as e:
        print(f"[错误] 无法保存图像: {e}")
    plt.close()


# ----------------------------------------------------------------------
# 步骤 3: 新的主函数 (main) - 用于推理
# ----------------------------------------------------------------------

# --- [ 关键修改 ] ---
def preprocess_image(image_path, map1, map2, roi, cfg, common_dims):
    """
    加载、矫正、并 *裁剪* 到最小公共 ROI
    (v8/v9 Bug 修复: 分离 gray 和 rgb)
    """
    img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        print(f"[错误] 无法加载图像: {image_path}")
        return None, None, None

    # 1. 矫正
    img_rect = cv2.remap(img_raw, map1, map2, cv2.INTER_LINEAR)

    # 2. 裁剪
    x, y, w, h = roi
    if y + h > img_rect.shape[0] or x + w > img_rect.shape[1] or y < 0 or x < 0:
        print(f"[错误] ROI {roi} 超出图像 {image_path} 边界 {img_rect.shape}")
        return None, None, None

    # 首先裁剪到各自的 ROI
    img_cropped_roi = img_rect[y:y + h, x:x + w]
    H_roi, W_roi = img_cropped_roi.shape[:2]

    # 3. [新] 再次裁剪到公共尺寸
    common_w, common_h = common_dims

    if H_roi < common_h or W_roi < common_w:
        print(f"[警告] ROI {roi} (尺寸 {W_roi}x{H_roi}) 小于计算的 common_dims {common_dims}。")
        common_h = min(H_roi, common_h)
        common_w = min(W_roi, common_w)

    img_cropped_common = img_cropped_roi[0:common_h, 0:common_w]

    # --- [ v8/v9 Bug 修复 ] ---
    # 1. 从 *原始* 图像创建 mask 和 gray_tensor
    _, mask = cv2.threshold(img_cropped_common, cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    img_gray_tensor = torch.from_numpy(img_cropped_common).float().unsqueeze(0) / 255.0
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

    # 2. *推理时不需要数据增强*
    # 3. 从 *原始* 图像创建 rgb_tensor
    img_rgb_tensor = torch.from_numpy(
        cv2.cvtColor(img_cropped_common, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0
    # --- [ 修复结束 ] ---

    # 返回张量
    return img_gray_tensor, img_rgb_tensor, mask_tensor


# --- [ 修改结束 ] ---


def main():
    parser = argparse.ArgumentParser(description="基于 Blob 检测器和鲁棒拟合的参数化波形重建")

    # --- [ v17 ] ---
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_sparse\20251116-004442\checkpoints\best_model_sparse.pth",
        help="路径：sparse_reconstructor_1115_gemini.py (v9) 训练保存的 .pth 模型文件。"
    )
    # --- [ v17 结束 ] ---

    parser.add_argument("--output_plot", type=str, default="reconstruction_blob_fit.png",
                        help="路径：保存拟合结果的可视化图像。")
    parser.add_argument("--min_disp", type=float, default=1.0, help="用于过滤的最小有效视差值。")

    # --- [ V17 修复 ] ---
    parser.add_argument("--confidence_threshold", type=float, default=0.2,
                        help="用于过滤的最小匹配置信度 (来自 Softmax)。")
    # --- [ V17 修复结束 ] ---

    parser.add_argument(
        "--dino_path",
        type=str,
        default="dinov3-base-model",
        help="DINO 模型的路径 (本地文件夹或HuggingFace ID)。"
    )
    parser.add_argument("--axis_height", type=int, default=1,
                        help="3D点云中代表“波高”的轴索引 (0=X, 1=Y, 2=Z)。默认为 1 (Y)。")
    parser.add_argument("--axis_propagation", type=int, default=2,
                        help="3D点云中代表“传播”的轴索引 (0=X, 1=Y, 2=Z)。默认为 2 (Z)。")
    parser.add_argument(
        "--filter_z_max",
        type=float,
        default=50000.0,
        help="过滤掉 Z 轴 (深度) 超过此值的点 (mm)。"
    )
    parser.add_argument("--filter_height_min", type=float, default=-500.0, help="过滤掉“波高”轴低于此值的点 (mm)。")
    parser.add_argument("--filter_height_max", type=float, default=500.0, help="过滤掉“波高”轴高于此值的点 (mm)。")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 创建配置 (v9)
    print(f"加载 v9 配置...")
    cfg = Config()
    cfg.DINO_LOCAL_PATH = args.dino_path

    # --- 自动加载路径 (与 1024 脚本相同) ---
    calibration_path = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                    "stereo_calib_params_from_matlab_full.npz")
    left_image_dir = os.path.join(DATA_ROOT, "data", "left_images")
    try:
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
        left_image_list = []
        for ext in extensions:
            left_image_list.extend(glob.glob(os.path.join(left_image_dir, f"left*{ext[1:]}")))
        if not left_image_list:
            for ext in extensions:
                left_image_list.extend(glob.glob(os.path.join(left_image_dir, ext)))
        if not left_image_list:
            raise FileNotFoundError(f"在 {left_image_dir} 中找不到任何图像文件。")
        left_image_list.sort()
        left_image_path = left_image_list[0]
    except Exception as e:
        print(f"[致命] 无法在 {left_image_dir} 中找到左图: {e}")
        sys.exit(1)
    right_image_dir = os.path.join(DATA_ROOT, "data", "right_images")
    left_basename = os.path.basename(left_image_path)
    if left_basename.startswith("left_"):
        right_filename = "right" + left_basename[4:]
    elif left_basename.startswith("left"):
        right_filename = "right" + left_basename[4:]
    else:
        print(f"[警告] 左图文件名 {left_basename} 不以 'left' 开头。假设右图在 {right_image_dir} 中同名。")
        right_filename = left_basename
    right_image_path = os.path.join(right_image_dir, right_filename)
    if not os.path.exists(right_image_path):
        if left_basename.startswith("left"):
            right_filename = "right_" + left_basename[4:]
            right_image_path = os.path.join(right_image_dir, right_filename)
    if not os.path.exists(right_image_path):
        print(f"[致命] 找到了左图 {left_image_path}，但未找到对应的右图（尝试了 {right_filename}）")
        sys.exit(1)
    print(f"--- 自动加载路径 ---")
    print(f"标定文件: {calibration_path}")
    print(f"  模型: {args.model_checkpoint}")
    print(f"  左图: {left_image_path}")
    print(f"  右图: {right_image_path}")
    print(f"--------------------")
    # --- 自动加载路径结束 ---

    # 2. 加载标定数据
    print(f"加载标定文件: {calibration_path}")
    Q, map1_l, map2_l, map1_r, map2_r, roi_l, roi_r = load_calibration(calibration_path)

    # --- [ 关键修改：计算公共 ROI ] ---
    common_w = min(roi_l[2], roi_r[2])
    common_h = min(roi_l[3], roi_r[3])
    common_dims = (common_w, common_h)
    print(f"计算的公共 ROI 尺寸 (WxH): {common_w} x {common_h}")
    # --- [ 修改结束 ] ---

    # 3. 加载模型
    print(f"加载模型定义 (V17: 完整 V9 模型)...")
    model = SparseMatchingStereoModel(cfg).to(device)

    # --- [ V17 关键修复 ] ---
    # 我们现在必须加载 V9 训练的检查点
    try:
        print(f"加载 V9 训练权重: {args.model_checkpoint}")
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=True)
    except Exception as e:
        print(f"[致命] 无法加载模型权重: {e}")
        print("       这可能是因为 V17 模型定义与 V9 检查点不匹配。")
        sys.exit(1)
    # --- [ V17 修复结束 ] ---
    model.eval()

    # 4. 加载和预处理图像
    print(f"预处理左图: {left_image_path}")
    l_gray, l_rgb, l_mask = preprocess_image(left_image_path, map1_l, map2_l, roi_l, cfg, common_dims)

    print(f"预处理右图: {right_image_path}")
    r_gray, r_rgb, _ = preprocess_image(right_image_path, map1_r, map2_r, roi_r, cfg, common_dims)

    if l_gray is None or r_gray is None:
        print("图像加载失败，退出。")
        sys.exit(1)

    # 5. 运行模型推理 (V17: 提取特征并匹配)
    print("运行模型推理 (V17: 提取特征并运行注意力匹配)...")
    with torch.no_grad():
        l_gray = l_gray.unsqueeze(0).to(device)
        r_gray = r_gray.unsqueeze(0).to(device)
        l_rgb = l_rgb.unsqueeze(0).to(device)
        r_rgb = r_rgb.unsqueeze(0).to(device)
        l_mask = l_mask.unsqueeze(0).to(device)

        patch_size = 16  # DINO patch size

        l_gray_pad, r_gray_pad, l_rgb_pad, r_rgb_pad, l_mask_pad = _pad_inputs(patch_size, l_gray, r_gray, l_rgb, r_rgb,
                                                                               l_mask)

        # 运行模型以获取 kp 和 desc
        outputs = model(l_gray_pad, r_gray_pad, l_rgb_pad, r_rgb_pad, l_mask_pad)

    print("推理完成。")

    # 6. 提取数据
    kp_left = outputs['keypoints_left'][0].cpu().numpy()
    scores_left = outputs['scores_left'][0].cpu().numpy()
    disparity_pred = outputs['disparity'][0].cpu().numpy()
    match_probs = outputs['match_probs'][0].cpu().numpy()  # [V17]

    # --- [ V17 关键修复 ] ---
    # 7. 基于置信度、视差和分数的过滤

    # 7.1. 计算每个左侧点的最高匹配置信度
    # (N_l, N_r) -> (N_l,)
    confidence = match_probs.max(axis=1)

    # 7.2. 定义过滤掩码
    valid_mask_score = scores_left > 0.1  # 必须是有效斑点
    valid_mask_disp = disparity_pred > args.min_disp  # 视差必须为正
    valid_mask_conf = confidence > args.confidence_threshold  # [V17] 置信度必须 > 阈值

    valid_mask = valid_mask_score & valid_mask_disp & valid_mask_conf

    kp_valid = kp_left[valid_mask]
    disp_valid = disparity_pred[valid_mask]

    print(
        f"V17 注意力匹配器: 找到 {np.sum(valid_mask)} 个有效匹配点 (score > 0.1, disp > {args.min_disp}, conf > {args.confidence_threshold})。")
    # --- [ V17 修改结束 ] ---

    # 8. 去除填充 (Padding) 区域的点
    H_common, W_common = common_dims[1], common_dims[0]
    if kp_valid.shape[0] > 0:
        valid_idx_mask = (kp_valid[:, 0] < W_common) & (kp_valid[:, 1] < H_common)
        kp_valid = kp_valid[valid_idx_mask]
        disp_valid = disp_valid[valid_idx_mask]
        print(f"去除填充区域: {np.sum(valid_idx_mask)} / {len(valid_idx_mask)} 个点")

    # --- [ V12 关键修复 ] ---
    # 9. 将坐标从 "ROI 裁剪空间" 转换回 "全矫正图像空间"
    #    Q 矩阵需要相对于全矫正图像的坐标
    roi_x, roi_y = roi_l[0], roi_l[1]
    kp_valid_rectified_space = kp_valid + np.array([roi_x, roi_y])
    print(f"V12 修复: 已将 {kp_valid.shape[0]} 个点坐标加上 ROI 偏移 (x+{roi_x}, y+{roi_y})")
    # --- [ V12 修复结束 ] ---

    # --- [ V13 关键修复: 修正视差 ] ---
    # 视差 d = u_L - u_R
    # 我们的 d (disp_valid) 是在 "公共裁剪空间" (common space) 中计算的: d_common = u_common_L - u_common_R
    # Q 矩阵期望的 d 是在 "完整矫正空间" (rectified space) 中计算的: d_Q = u_rect_L - u_rect_R
    # u_rect_L = u_common_L + roi_l[0]
    # u_rect_R = u_common_R + roi_r[0]
    # d_Q = (u_common_L + roi_l[0]) - (u_common_R + roi_r[0])
    # d_Q = (u_common_L - u_common_R) + (roi_l[0] - roi_r[0])
    # d_Q = d_common + (roi_l[0] - roi_r[0])

    disparity_roi_offset = roi_l[0] - roi_r[0]
    disp_valid_rectified_space = disp_valid + disparity_roi_offset

    print(f"V1T 修复: 视差已修正。 ROI 偏移 (L_x - R_x): {disparity_roi_offset}")
    if disp_valid.shape[0] > 0:
        print(
            f"         原始平均视差: {np.mean(disp_valid):.2f}, 修正后平均视差: {np.mean(disp_valid_rectified_space):.2f}")
    # --- [ V13 修复结束 ] ---

    # 10. 重投影到 3D
    print("重投影到 3D 空间...")
    points_3d_raw = reproject_to_3d(kp_valid_rectified_space, disp_valid_rectified_space, Q)  # <-- [V13] 传入修复后的视差

    # 11. 过滤 3D 点云
    points_3d_filtered = filter_point_cloud(
        points_3d_raw,
        height_axis=args.axis_height,
        prop_axis=args.axis_propagation,
        z_max=args.filter_z_max,
        height_range=(args.filter_height_min, args.filter_height_max)
    )

    if points_3d_filtered.shape[0] == 0:
        print("[错误] 3D 点云过滤后没有剩余的点。退出。")
        return

    # 12. 拟合参数 (使用鲁棒拟合)
    fit_params, _, inlier_mask = fit_wave_parameters(
        points_3d_filtered,
        height_axis=args.axis_height,
        prop_axis=args.axis_propagation
    )

    if fit_params is not None:
        A, k, phi, y_offset = fit_params
        Amplitude = A * 2
        Wavelength = (2 * np.pi) / (k + 1e-9)
        print("\n--- 重建结果 ---")
        print(f"  拟合波幅 (A): {A:.4f} mm")
        print(f"  拟合波高 (2*A): {Amplitude:.4f} mm")
        print(f"  拟合波数 (k): {k:.6f} rad/mm")
        print(f"  拟合波长 (L): {Wavelength:.2f} mm")
        print(f"  拟合相位 (phi): {phi:.4f} rad")
        print(f"  拟合偏移 (offset): {y_offset:.4f} mm")
        print("--------------------")

    # 13. 可视化
    plot_reconstruction(
        points_3d_filtered,
        fit_params,
        inlier_mask,
        args.output_plot,
        height_axis=args.axis_height,
        prop_axis=args.axis_propagation,
        common_dims=common_dims  # [v9] 传入公共尺寸
    )


if __name__ == "__main__":
    main()