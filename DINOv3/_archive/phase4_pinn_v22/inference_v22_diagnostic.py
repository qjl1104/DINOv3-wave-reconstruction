import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import glob

# --- 检查 transformers ---
try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80 + "\n[FATAL] transformers not found. pip install transformers\n" + "=" * 80)
    sys.exit(1)


# --- 配置类 ---
@dataclass
class Config:
    DINO_LOCAL_PATH: str = "dinov3-base-model"
    MAX_KEYPOINTS: int = 1024
    FEATURE_DIM: int = 768
    NUM_ATTENTION_LAYERS: int = 6
    NUM_HEADS: int = 8
    MATCHING_TEMPERATURE: float = 15.0
    MASK_THRESHOLD: int = 30
    BLOB_MIN_THRESHOLD: float = 15.0
    BLOB_MIN_AREA: float = 10.0
    BLOB_MAX_AREA: float = 2500.0


# ----------------------------------------------------------------------
# 1. 模型定义 (不变)
# ----------------------------------------------------------------------
class SparseKeypointDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_k = cfg.MAX_KEYPOINTS
        p = cv2.SimpleBlobDetector_Params()
        p.filterByColor = False
        p.minThreshold = cfg.BLOB_MIN_THRESHOLD
        p.maxThreshold = 255
        p.filterByArea = True
        p.minArea = cfg.BLOB_MIN_AREA
        p.maxArea = cfg.BLOB_MAX_AREA
        p.filterByCircularity = True
        p.minCircularity = 0.1
        p.filterByConvexity = True
        p.minConvexity = 0.85
        p.filterByInertia = True
        p.minInertiaRatio = 0.1
        self.det = cv2.SimpleBlobDetector_create(p)

    def forward(self, img, mask):
        B = img.shape[0]
        kpts, scores = [], []
        for b in range(B):
            im_np = (img[b, 0].cpu().numpy() * 255).astype(np.uint8)
            kps = self.det.detect(im_np)
            if not kps:
                kpts.append(torch.zeros(1, 2, device=img.device))
                scores.append(torch.zeros(1, device=img.device))
                continue
            pts = np.array([k.pt for k in kps]).astype(np.float32)
            sz = np.array([k.size for k in kps]).astype(np.float32)
            pt_t = torch.from_numpy(pts).to(img.device)
            sz_t = torch.from_numpy(sz).to(img.device)
            if len(pt_t) > self.max_k:
                idx = torch.argsort(sz_t, descending=True)[:self.max_k]
                pt_t = pt_t[idx]
                sz_t = sz_t[idx]
            kpts.append(pt_t)
            scores.append(sz_t)
        max_l = max([len(k) for k in kpts])
        if max_l == 0: max_l = 1
        k_pad, s_pad = [], []
        for k, s in zip(kpts, scores):
            pad_n = max_l - len(k)
            if pad_n > 0:
                k = torch.cat([k, torch.zeros(pad_n, 2, device=img.device)], 0)
                s = torch.cat([s, torch.zeros(pad_n, device=img.device)], 0)
            k_pad.append(k)
            s_pad.append(s)
        return torch.stack(k_pad), torch.stack(s_pad)


class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            self.dino = AutoModel.from_pretrained(cfg.DINO_LOCAL_PATH, local_files_only=True)
        except:
            print(f"Warning: Local DINO not found, trying default.")
            self.dino = AutoModel.from_pretrained("facebook/dinov2-base")
        for p in self.dino.parameters(): p.requires_grad = False
        self.feat_dim = self.dino.config.hidden_size
        self.patch = self.dino.config.patch_size

    def forward(self, img, kpts):
        with torch.no_grad():
            out = self.dino(img).last_hidden_state
        B, _, H, W = img.shape
        n_patches_h = H // self.patch
        n_patches_w = W // self.patch
        feat = out[:, -(n_patches_h * n_patches_w):]
        feat = feat.transpose(1, 2).reshape(B, self.feat_dim, n_patches_h, n_patches_w)
        grid = kpts.clone()
        grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
        grid = grid.unsqueeze(2)
        desc = F.grid_sample(feat, grid, align_corners=True, padding_mode='border')
        return desc.squeeze(3).transpose(1, 2)


class SparseMatchingStereoModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.det = SparseKeypointDetector(cfg)
        self.ext = DINOv3FeatureExtractor(cfg)
        self.proj = nn.Linear(2, cfg.FEATURE_DIM)
        layer = nn.TransformerEncoderLayer(d_model=cfg.FEATURE_DIM, nhead=cfg.NUM_HEADS, batch_first=True)
        self.trans = nn.TransformerEncoder(layer, num_layers=cfg.NUM_ATTENTION_LAYERS)
        self.out_proj = nn.Linear(cfg.FEATURE_DIM, cfg.FEATURE_DIM)

    def forward(self, lg, rg, lrgb, rrgb, mask):
        kpl, sl = self.det(lg, mask)
        kpr, sr = self.det(rg, torch.ones_like(rg))
        descl = self.ext(lrgb, kpl)
        descr = self.ext(rrgb, kpr)
        B, N, _ = kpl.shape
        H, W = lg.shape[2:]
        posl = self.proj(kpl / max(H, W))
        posr = self.proj(kpr / max(H, W))
        featl = self.trans(descl + posl)
        featr = self.trans(descr + posr)
        featl = F.normalize(self.out_proj(featl), dim=-1)
        featr = F.normalize(self.out_proj(featr), dim=-1)
        scores = torch.bmm(featl, featr.transpose(1, 2)) * self.cfg.MATCHING_TEMPERATURE
        mask_geo = (kpl[:, :, 1].unsqueeze(2) - kpr[:, :, 1].unsqueeze(1)).abs() < 2.0
        scores = scores.masked_fill(~mask_geo, -1e9)
        probs = F.softmax(scores, dim=-1)
        x_right_ex = (probs * kpr[:, :, 0].unsqueeze(1)).sum(dim=2)
        disp = kpl[:, :, 0] - x_right_ex
        return {'keypoints_left': kpl, 'scores_left': sl, 'disparity': disp}


# ----------------------------------------------------------------------
# 2. 坐标变换核心 (只用 PCA 找平，不缩放)
# ----------------------------------------------------------------------

def align_point_cloud_pca(points):
    if len(points) < 3: return points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    u, s, vt = np.linalg.svd(centered)
    normal = vt[2, :]
    if normal[1] < 0: normal = -normal

    target = np.array([0, 1, 0])
    v = np.cross(normal, target)
    c = np.dot(normal, target)
    s_val = np.linalg.norm(v)

    if s_val < 1e-6:
        R_mat = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_mat = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s_val ** 2))

    aligned = (R_mat @ centered.T).T
    return aligned


# ----------------------------------------------------------------------
# 3. 诊断性拟合函数 (不做任何 Scale)
# ----------------------------------------------------------------------

def wave_func(z, A, k, phi, offset):
    return A * np.cos(k * z + phi) + offset


def fit_wave_diagnostic(z, y):
    """
    完全不带预设 Scale 的拟合，只为了看原始数据到底长什么样
    """
    # 宽松的初始猜测
    A_guess = (np.max(y) - np.min(y)) / 2
    k_guess = 2 * np.pi / 2500
    p0 = [A_guess, k_guess, 0, np.mean(y)]

    # 不设 bounds，看看它自己想拟合成什么样
    try:
        popt, _ = curve_fit(wave_func, z, y, p0=p0, maxfev=20000)
    except:
        return None, None

    y_pred = wave_func(z, *popt)
    res = np.abs(y - y_pred)

    # 宽松的 RANSAC 阈值
    mask = res < 30.0

    if mask.sum() < 10: return None, None

    try:
        popt_final, _ = curve_fit(wave_func, z[mask], y[mask], p0=popt, maxfev=20000)
    except:
        return None, None

    return popt_final, mask


# ----------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------

def load_calibration(calib_file):
    data = np.load(calib_file)
    return (data['Q'],
            data['map1_left'], data['map2_left'],
            data['map1_right'], data['map2_right'],
            tuple(map(int, data['roi_left'])),
            tuple(map(int, data['roi_right'])))


def preprocess_image(path, map1, map2, roi, mask_thresh, target_size=None):
    img = cv2.imread(path, 0)
    if img is None: return None, None, None
    img_rect = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    x, y, w, h = roi
    img_crop = img_rect[y:y + h, x:x + w]
    if target_size:
        h_t, w_t = target_size
        img_crop = img_crop[:h_t, :w_t]
    _, mask = cv2.threshold(img_crop, mask_thresh, 255, cv2.THRESH_BINARY)
    g_tensor = torch.from_numpy(img_crop).float().unsqueeze(0) / 255.0
    m_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
    rgb = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2RGB)
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return g_tensor, rgb_tensor, m_tensor


def reproject_to_3d_optimized(kpts, disp, Q, roi_l, roi_r):
    u_L = kpts[:, 0] + roi_l[0]
    v_L = kpts[:, 1] + roi_l[1]
    roi_shift = roi_l[0] - roi_r[0]
    d_corrected = disp + roi_shift
    points_vec = np.stack([u_L, v_L, d_corrected, np.ones_like(d_corrected)], axis=1)
    homog_points = (Q @ points_vec.T).T
    w = homog_points[:, 3]
    valid = np.abs(w) > 1e-6
    X = homog_points[valid, 0] / w[valid]
    Y = homog_points[valid, 1] / w[valid]
    Z = homog_points[valid, 2] / w[valid]
    return np.stack([X, Y, Z], axis=1)


def main():
    # --- 路径 ---
    checkpoint_path = r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_physics_V20\20251119-172550\checkpoints\model_ep150.pth"
    dino_path = r"D:\Research\wave_reconstruction_project\DINOv3\dinov3-base-model"
    calib_path = r"D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_from_matlab_full.npz"
    left_img_dir = r"D:\Research\wave_reconstruction_project\data\left_images"
    right_img_dir = r"D:\Research\wave_reconstruction_project\data\right_images"

    CONF_THRESH = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = Config()
    cfg.DINO_LOCAL_PATH = dino_path
    model = SparseMatchingStereoModel(cfg).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    Q, m1l, m2l, m1r, m2r, roi_l, roi_r = load_calibration(calib_path)

    l_files = sorted(glob.glob(os.path.join(left_img_dir, "left*.*")))
    if not l_files: return

    l_path = l_files[0]
    basename = os.path.basename(l_path)
    r_basename = "right" + basename[4:] if basename.startswith("left") else basename.replace("left", "right")
    r_path = os.path.join(right_img_dir, r_basename)

    print(f"Processing: {basename}")

    target_w = min(roi_l[2], roi_r[2])
    target_h = min(roi_l[3], roi_r[3])

    lg, lrgb, lm = preprocess_image(l_path, m1l, m2l, roi_l, cfg.MASK_THRESHOLD, (target_h, target_w))
    rg, rrgb, _ = preprocess_image(r_path, m1r, m2r, roi_r, cfg.MASK_THRESHOLD, (target_h, target_w))

    def pad14(t):
        h, w = t.shape[-2:]
        ph = (14 - h % 14) % 14
        pw = (14 - w % 14) % 14
        return F.pad(t, (0, pw, 0, ph))

    lg, rg, lrgb, rrgb, lm = map(pad14, [lg, rg, lrgb, rrgb, lm])
    inputs = [t.unsqueeze(0).to(device) for t in [lg, rg, lrgb, rrgb, lm]]

    with torch.no_grad():
        out = model(*inputs)

    kpl = out['keypoints_left'][0].cpu().numpy()
    scores = out['scores_left'][0].cpu().numpy()
    disp = out['disparity'][0].cpu().numpy()

    mask_final = (scores > CONF_THRESH) & (disp > 1.0)
    points_3d = reproject_to_3d_optimized(kpl[mask_final], disp[mask_final], Q, roi_l, roi_r)

    print(f"Raw Valid Points: {len(points_3d)}")

    # 1. 自动找平 (PCA) - 唯一允许的几何修正
    mask_dist = (points_3d[:, 2] > 500) & (points_3d[:, 2] < 15000)
    points_aligned = align_point_cloud_pca(points_3d[mask_dist])

    # 2. 物理过滤 (非常宽松)
    mask_height = np.abs(points_aligned[:, 1]) < 500
    points_clean = points_aligned[mask_height]

    if len(points_clean) < 10:
        print("Too few points.")
        return

    range_x = points_clean[:, 0].max() - points_clean[:, 0].min()
    range_z = points_clean[:, 2].max() - points_clean[:, 2].min()
    if range_x > range_z:
        Z_data = points_clean[:, 0]
        X_data = points_clean[:, 2]
    else:
        Z_data = points_clean[:, 2]
        X_data = points_clean[:, 0]
    Y_data = points_clean[:, 1]

    idx_sort = np.argsort(Z_data)
    Z_data = Z_data[idx_sort]
    Y_data = Y_data[idx_sort]

    # --- 3. 纯诊断拟合 (Pure Diagnostic Fit) ---
    popt, inlier_mask = fit_wave_diagnostic(Z_data, Y_data)

    print("\n" + "=" * 40)
    print("   纯诊断结果 (无缩放，无假设)   ")
    print("=" * 40)
    if popt is not None:
        A, k, phi, offset = popt
        wavelength = 2 * np.pi / k
        height_val = abs(A) * 2

        print(f"实测波高 (H): {height_val:.2f} mm (理论值: 80mm)")
        print(f"实测波长 (L): {wavelength:.2f} mm (理论值: 2500mm)")

        # 智能分析
        if abs(height_val - 80.0) > 40.0:
            print("\n[分析结论] 波高严重偏离！")
            print("可能原因：")
            print("1. 基线 (Baseline) 标定值 T 有误。双目视差 d = f * T / Z。T 错了，Z 就会整体缩放。")
            print("2. 焦距 (f) 标定有误。")
            print("建议：检查 Q 矩阵中的 Q[2,3] (f) 和 Q[3,2] (1/T)。")
        else:
            print("\n[分析结论] 结果非常准确！之前的'不干净'可能只是图画得不对。")
    else:
        print("拟合失败，点云分布过于杂乱，无法提取波形。")

    # 可视化
    fig = plt.figure(figsize=(14, 6))
    plt.rcParams['axes.unicode_minus'] = False

    # 3D
    ax = fig.add_subplot(121, projection='3d')
    sc = ax.scatter(X_data, Z_data, Y_data, c=Y_data, cmap='jet', s=15, alpha=0.9)
    ax.set_xlabel('Width X')
    ax.set_ylabel('Propagation Z')
    ax.set_zlabel('Height Y')
    ax.set_title(f'Raw Aligned Point Cloud')
    plt.colorbar(sc, ax=ax, label='Height (mm)', shrink=0.5)
    ax.view_init(elev=20, azim=-70)

    # 拟合
    ax2 = fig.add_subplot(122)
    ax2.scatter(Z_data, Y_data, c='gray', alpha=0.3, s=15, label='Raw Points')
    if popt is not None:
        ax2.scatter(Z_data[inlier_mask], Y_data[inlier_mask], c='blue', alpha=0.6, s=15, label='Inliers')
        z_fit = np.linspace(Z_data.min(), Z_data.max(), 500)
        y_fit = wave_func(z_fit, *popt)
        label_str = f'Fit: H={height_val:.0f}mm'
        ax2.plot(z_fit, y_fit, 'r-', linewidth=2, label=label_str)

    ax2.set_title('Diagnostic Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('result_v25_diagnostic.png', dpi=300)
    print("\nResult saved to: result_v25_diagnostic.png")


if __name__ == "__main__":
    main()