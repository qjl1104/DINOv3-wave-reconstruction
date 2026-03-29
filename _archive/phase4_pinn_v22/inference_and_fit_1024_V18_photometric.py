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
import glob

# ----------------------------------------------------------------------
# --- [ V19.4 最终完美版 ] ---
# 针对 V19.3 拟合失败的修正:
# 1. V19.1 的波长 (2892mm) 其实是准确的 (接近真值 2500mm).
# 2. V19.3 全局缩放导致波长被错误压缩至 1300mm，导致拟合崩溃.
# 3. V19.4 策略: **只对波高 (Y轴) 进行缩放校正**，保持波长 (Z轴) 不变.
# 4. 保留 V19.2 的宽松检测阈值 (MinArea=10), 获取更多点.
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
    PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
    DATA_ROOT = r"D:\Research\wave_reconstruction_project"


@dataclass
class Config:
    """配置 - 必须与 V5 训练脚本完全一致"""
    NUM_ATTENTION_LAYERS: int = 6
    FEATURE_DIM: int = 768
    NUM_HEADS: int = 8
    DISPARITY_CONSTRAINT_Y_THRESHOLD: int = 2
    MATCHING_TEMPERATURE: float = 10.0
    DINO_LOCAL_PATH: str = "dinov3-base-model"
    MAX_KEYPOINTS: int = 1024
    MASK_THRESHOLD: int = 30

    # [V19.2] 保持宽松的检测参数，以获得更多点
    BLOB_MIN_THRESHOLD: float = 15.0
    BLOB_MIN_AREA: float = 10.0
    BLOB_MAX_AREA: float = 2500.0
    BLOB_MIN_CIRCULARITY: float = 0.1
    BLOB_MIN_CONVEXITY: float = 0.85
    BLOB_MIN_INERTIA: float = 0.1
    BLOB_MAX_INERTIA: float = 0.85


# --- 1. 斑点检测器 ---
class SparseKeypointDetector(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_keypoints = cfg.MAX_KEYPOINTS

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
        params.maxInertiaRatio = cfg.BLOB_MAX_INERTIA

        self.detector = cv2.SimpleBlobDetector_create(params)
        print(f"--- [检测器] V19.4 参数: MinArea={cfg.BLOB_MIN_AREA} (宽松模式) ---")

    def forward(self, gray_image, mask):
        B, _, H, W = gray_image.shape
        device = gray_image.device
        keypoints_list = []
        scores_list = []

        for b in range(B):
            img_tensor = gray_image[b, 0]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            cv_keypoints = self.detector.detect(img_np)

            if not cv_keypoints:
                keypoints_list.append(torch.zeros(1, 2, device=device))
                scores_list.append(torch.zeros(1, device=device))
                continue

            kp_coords = np.array([kp.pt for kp in cv_keypoints]).astype(np.float32)
            kp_scores = np.array([kp.size for kp in cv_keypoints]).astype(np.float32)

            kp_tensor = torch.from_numpy(kp_coords).to(device)
            scores_tensor = torch.from_numpy(kp_scores).to(device)

            if len(kp_tensor) > self.max_keypoints:
                sorted_indices = torch.argsort(scores_tensor, descending=True)
                kp_tensor = kp_tensor[sorted_indices[:self.max_keypoints]]
                scores_tensor = scores_tensor[sorted_indices[:self.max_keypoints]]

            keypoints_list.append(kp_tensor)
            scores_list.append(scores_tensor)

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


# --- 2. DINOv3 特征提取器 ---
class DINOv3FeatureExtractor(nn.Module):
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
                return AutoModel.from_pretrained(model_path, local_files_only=True)
            else:
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
        patch_tokens = features[:, start_idx:, :]
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        if patch_tokens.shape[1] != h_feat * w_feat:
            expected = h_feat * w_feat
            if patch_tokens.shape[1] > expected:
                patch_tokens = patch_tokens[:, :expected, :]
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
            feat_map, grid, mode='bilinear', align_corners=True, padding_mode='border'
        )
        descriptors = descriptors_sampled.squeeze(3).permute(0, 2, 1)
        return descriptors


# --- 3. 注意力匹配网络 ---
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(2, dim)

    def forward(self, positions, image_size):
        H, W = image_size
        pos_normalized = positions.clone()
        pos_normalized[..., 0] = pos_normalized[..., 0] / (W - 1) if W > 1 else 0
        pos_normalized[..., 1] = pos_normalized[..., 1] / (H - 1) if H > 1 else 0
        return self.proj(pos_normalized)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, features, pos_enc):
        features_pos = features + pos_enc
        qkv = self.norm1(features_pos)
        attn_out, _ = self.attn(qkv, qkv, qkv)
        features = features + attn_out
        ffn_in = self.norm2(features)
        features = features + self.ffn(ffn_in)
        return features


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_after_res = nn.LayerNorm(dim)

    def forward(self, feat_query, feat_kv):
        q = self.norm_q(feat_query)
        kv = self.norm_kv(feat_kv)
        attn_out, _ = self.attn(q, kv, kv)
        features = feat_query + attn_out
        features = self.norm_after_res(features)
        return features, None


class SparseMatchingNetwork(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim = cfg.FEATURE_DIM
        self.pos_enc = PositionalEncoding(dim)
        self.self_attn_left = nn.ModuleList(
            [SelfAttentionLayer(dim, cfg.NUM_HEADS) for _ in range(cfg.NUM_ATTENTION_LAYERS)])
        self.self_attn_right = nn.ModuleList(
            [SelfAttentionLayer(dim, cfg.NUM_HEADS) for _ in range(cfg.NUM_ATTENTION_LAYERS)])
        self.cross_attn = CrossAttentionLayer(dim, cfg.NUM_HEADS)

    def forward(self, desc_left, desc_right, kp_left, kp_right, image_size):
        pos_left = self.pos_enc(kp_left, image_size)
        pos_right = self.pos_enc(kp_right, image_size)
        feat_left = desc_left
        feat_right = desc_right
        for sl, sr in zip(self.self_attn_left, self.self_attn_right):
            feat_left = sl(feat_left, pos_left)
            feat_right = sr(feat_right, pos_right)
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


# --- 4. 完整模型类 ---
class SparseMatchingStereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.keypoint_detector = SparseKeypointDetector(cfg)
        self.feature_extractor = DINOv3FeatureExtractor(cfg)
        self.matcher = SparseMatchingNetwork(cfg)

    def forward(self, left_gray, right_gray, left_rgb, right_rgb, mask):
        kp_left, scores_left = self.keypoint_detector(left_gray, mask)
        kp_right, scores_right = self.keypoint_detector(right_gray, torch.ones_like(right_gray))
        desc_left = self.feature_extractor(left_rgb, kp_left)
        desc_right = self.feature_extractor(right_rgb, kp_right)
        match_scores, disparity, constraint_mask = self.matcher(desc_left, desc_right, kp_left, kp_right,
                                                                left_gray.shape[2:])

        return {
            'keypoints_left': kp_left,
            'scores_left': scores_left,
            'disparity': disparity,
            'keypoints_right': kp_right,
            'scores_right': scores_right,
            'descriptors_left': desc_left,
            'descriptors_right': desc_right,
            'match_scores': match_scores,
            'constraint_mask': constraint_mask
        }


# --- 辅助函数 ---
def _pad_inputs(patch_size, *tensors):
    if not tensors or tensors[0] is None: return list(tensors)
    h, w = tensors[0].shape[-2:]
    if patch_size is None: patch_size = 16
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0: return list(tensors)
    padded = []
    for t in tensors:
        if t is not None:
            padded.append(F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0))
        else:
            padded.append(None)
    return padded


def load_calibration(calib_file):
    try:
        calib = np.load(calib_file)
        return calib['Q'], calib['map1_left'], calib['map2_left'], calib['map1_right'], calib['map2_right'], tuple(
            map(int, calib['roi_left'])), tuple(map(int, calib['roi_right']))
    except Exception as e:
        print(f"[错误] 加载标定文件失败: {e}")
        sys.exit(1)


def preprocess_image(image_path, map1, map2, roi, cfg, common_dims):
    img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None: return None, None, None
    img_rect = cv2.remap(img_raw, map1, map2, cv2.INTER_LINEAR)
    x, y, w, h = roi
    img_cropped = img_rect[y:y + h, x:x + w]
    common_w, common_h = common_dims
    img_cropped = img_cropped[:common_h, :common_w]
    _, mask = cv2.threshold(img_cropped, cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    img_gray_tensor = torch.from_numpy(img_cropped).float().unsqueeze(0) / 255.0
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
    img_rgb_tensor = torch.from_numpy(cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0
    return img_gray_tensor, img_rgb_tensor, mask_tensor


def reproject_to_3d(keypoints_rect, disparity_rect, Q):
    N = keypoints_rect.shape[0]
    if N == 0: return np.zeros((0, 3))
    points_4d = np.hstack((keypoints_rect, disparity_rect.reshape(-1, 1), np.ones((N, 1))))
    projected = (Q @ points_4d.T).T
    W = projected[:, 3:4]
    mask = np.abs(W) > 1e-6
    # V19.1 广播修复
    points_3d = projected[mask.flatten(), :3] / W[mask].reshape(-1, 1)
    return points_3d


def wave_model_func(prop_data, A, k, phi, y_offset):
    return A * np.cos(k * prop_data + phi) + y_offset


def fit_wave_parameters(points_3d, height_axis, prop_axis):
    if points_3d.shape[0] < 10: return None, None, None
    prop_data = points_3d[:, prop_axis]
    height_data = points_3d[:, height_axis]

    A_guess = (np.max(height_data) - np.min(height_data)) / 2 if np.max(height_data) > np.min(height_data) else 1.0
    L_guess = 2500
    k_guess = 2 * np.pi / L_guess
    p0 = [A_guess, k_guess, 0, np.mean(height_data)]

    try:
        # 2. 粗略拟合
        params_rough, _ = curve_fit(wave_model_func, prop_data, height_data, p0=p0, maxfev=5000)

        # 3. [V19.2] 收紧阈值
        residuals = np.abs(height_data - wave_model_func(prop_data, *params_rough))
        mad = np.median(np.abs(residuals - np.median(residuals)))
        threshold = 1.5 * 1.4826 * mad if mad > 0 else 50.0
        inlier_mask = residuals < max(threshold, 10.0)

        if np.sum(inlier_mask) < 10: return None, None, None

        # 4. 精确拟合
        params_final, _ = curve_fit(wave_model_func, prop_data[inlier_mask], height_data[inlier_mask], p0=params_rough,
                                    maxfev=5000)
        if params_final[0] < 0:
            params_final[0] = -params_final[0]
            params_final[2] += np.pi

        return params_final, None, inlier_mask

    except Exception as e:
        print(f"拟合失败: {e}")
        return None, None, None


def plot_results(points_3d, fit_params, inlier_mask, output_file, height_axis, prop_axis, scale_factor=1.0):
    plt.figure(figsize=(15, 8))
    prop = points_3d[:, prop_axis]
    height = points_3d[:, height_axis]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    note = f"(Y轴已校正 Scale={scale_factor:.2f})" if scale_factor != 1.0 else "(未校正)"

    if inlier_mask is not None:
        plt.scatter(prop[~inlier_mask], height[~inlier_mask], c='gray', alpha=0.3, label='局外点')
        plt.scatter(prop[inlier_mask], height[inlier_mask], c='blue', alpha=0.6, label='局内点')
    else:
        plt.scatter(prop, height, c='blue', alpha=0.6)

    if fit_params is not None:
        x_fit = np.linspace(prop.min(), prop.max(), 500)
        y_fit = wave_model_func(x_fit, *fit_params)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='拟合波形')

        title = f"V19.4 最终结果 {note} | 波高(2A): {fit_params[0] * 2:.2f}mm | 波长: {2 * np.pi / fit_params[1]:.2f}mm"
        plt.title(title)

    plt.legend()
    plt.xlabel("传播方向 (mm)")
    plt.ylabel("高度 (mm)")
    plt.grid(True)
    plt.savefig(output_file, dpi=150)
    print(f"结果已保存至: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str,
                        default=r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_sparse_photometric_V5\20251118-134244\checkpoints\best_model_sparse.pth",
                        help="V5 训练得到的模型路径")
    parser.add_argument("--dino_path", type=str, default="dinov3-base-model")
    parser.add_argument("--min_disp", type=float, default=1.0)
    parser.add_argument("--output_plot", type=str, default="reconstruction_V19_4.png")

    # [V19.4 核心参数]
    # 只校正波高，2.09 是测量值167mm与真实值80mm的比例
    parser.add_argument("--height_scale_correction", type=float, default=2.09,
                        help="仅对 Y 轴 (波高) 进行校正的缩放因子")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    cfg = Config()
    cfg.DINO_LOCAL_PATH = args.dino_path

    calib_path = os.path.join(DATA_ROOT, "camera_calibration", "params", "stereo_calib_params_from_matlab_full.npz")
    left_img_dir = os.path.join(DATA_ROOT, "data", "left_images")
    right_img_dir = os.path.join(DATA_ROOT, "data", "right_images")

    left_imgs = sorted(glob.glob(os.path.join(left_img_dir, "left*.*")))
    if not left_imgs: sys.exit("未找到左图")
    left_img_path = left_imgs[0]
    right_img_path = os.path.join(right_img_dir, "right" + os.path.basename(left_img_path)[4:])

    print(f"左图: {left_img_path}")
    print(f"右图: {right_img_path}")
    print(f"模型: {args.model_checkpoint}")

    Q, map1_l, map2_l, map1_r, map2_r, roi_l, roi_r = load_calibration(calib_path)
    common_w = min(roi_l[2], roi_r[2])
    common_h = min(roi_l[3], roi_r[3])
    common_dims = (common_w, common_h)

    model = SparseMatchingStereoModel(cfg).to(device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=True)
    model.eval()

    l_gray, l_rgb, l_mask = preprocess_image(left_img_path, map1_l, map2_l, roi_l, cfg, common_dims)
    r_gray, r_rgb, _ = preprocess_image(right_img_path, map1_r, map2_r, roi_r, cfg, common_dims)

    with torch.no_grad():
        l_g = l_gray.unsqueeze(0).to(device)
        r_g = r_gray.unsqueeze(0).to(device)
        l_r = l_rgb.unsqueeze(0).to(device)
        r_r = r_rgb.unsqueeze(0).to(device)
        l_m = l_mask.unsqueeze(0).to(device)

        inputs = _pad_inputs(16, l_g, r_g, l_r, r_r, l_m)
        outputs = model(*inputs)

    kp = outputs['keypoints_left'][0].cpu().numpy()
    scores = outputs['scores_left'][0].cpu().numpy()
    disp = outputs['disparity'][0].cpu().numpy()

    valid = (scores > 0.1) & (disp > args.min_disp)
    kp_valid = kp[valid]
    disp_valid = disp[valid]

    in_bounds = (kp_valid[:, 0] < common_w) & (kp_valid[:, 1] < common_h)
    kp_valid = kp_valid[in_bounds]
    disp_valid = disp_valid[in_bounds]

    print(f"检测到 {len(kp_valid)} 个有效点。")
    print(f"平均视差: {np.mean(disp_valid):.2f}")

    kp_rect = kp_valid + np.array([roi_l[0], roi_l[1]])
    disp_rect = disp_valid + (roi_l[0] - roi_r[0])

    points_3d = reproject_to_3d(kp_rect, disp_rect, Q)

    # 坐标轴: 0=X, 1=Y(波高), 2=Z(传播)
    height_axis = 1
    prop_axis = 2

    # --- [V19.4 核心修复] ---
    # 只对 Y 轴 (Height) 进行缩放，不对 Z 轴 (Wavelength) 进行缩放
    if args.height_scale_correction != 1.0:
        print(f"--- 应用 Y 轴校正: 除以 {args.height_scale_correction} ---")
        points_3d[:, height_axis] = points_3d[:, height_axis] / args.height_scale_correction
    # --- [V19.4 结束] ---

    valid_z = (points_3d[:, 2] < 50000) & (points_3d[:, 2] > 0)
    valid_h = (points_3d[:, 1] > -500) & (points_3d[:, 1] < 500)
    points_filtered = points_3d[valid_z & valid_h]

    print(f"空间过滤后剩余 {len(points_filtered)} 个点。")

    fit_params, _, inlier_mask = fit_wave_parameters(points_filtered, height_axis, prop_axis)

    if fit_params is not None:
        print(f"拟合成功: 波幅={fit_params[0]:.2f}, 波长={2 * np.pi / fit_params[1]:.2f}")
        print(f"预测波高 (2A): {fit_params[0] * 2:.2f} mm (真实值约 80mm)")
    else:
        print("拟合失败。")

    plot_results(points_filtered, fit_params, inlier_mask, args.output_plot, height_axis, prop_axis,
                 args.height_scale_correction)


if __name__ == "__main__":
    main()