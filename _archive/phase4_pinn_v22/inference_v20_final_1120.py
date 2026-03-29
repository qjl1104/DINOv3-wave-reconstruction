import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import torch
import torch.nn.functional as F
import os
import sys
from dataclasses import dataclass
import glob
import warnings

# --- 配置区域 ---
warnings.filterwarnings("ignore")


@dataclass
class Config:
    # 路径配置
    DINO_LOCAL_PATH: str = "dinov3-base-model"
    CHECKPOINT_PATH: str = r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_physics_V20\20251119-172550\checkpoints\model_ep150.pth"
    CALIB_PATH: str = r"D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_from_matlab_full.npz"
    LEFT_IMG_DIR: str = r"D:\Research\wave_reconstruction_project\data\left_images"
    RIGHT_IMG_DIR: str = r"D:\Research\wave_reconstruction_project\data\right_images"

    # 算法参数
    MAX_KEYPOINTS: int = 1024
    FEATURE_DIM: int = 768
    NUM_ATTENTION_LAYERS: int = 6
    NUM_HEADS: int = 8

    MATCHING_TEMPERATURE: float = 15.0
    CONF_THRESH: float = 0.2
    MASK_THRESHOLD: int = 30
    BLOB_MIN_THRESHOLD: float = 15.0
    BLOB_MIN_AREA: float = 10.0


# --- 模型组件 ---
try:
    from transformers import AutoModel
except ImportError:
    sys.exit("Please install transformers: pip install transformers")


class SparseKeypointDetector(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_k = cfg.MAX_KEYPOINTS
        p = cv2.SimpleBlobDetector_Params()
        p.filterByColor = False
        p.minThreshold = cfg.BLOB_MIN_THRESHOLD
        p.maxThreshold = 255
        p.filterByArea = True
        p.minArea = cfg.BLOB_MIN_AREA
        p.maxArea = 2500.0
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
        k_pad = [torch.cat([k, torch.zeros(max_l - len(k), 2, device=img.device)], 0) for k in kpts]
        s_pad = [torch.cat([s, torch.zeros(max_l - len(s), device=img.device)], 0) for s in scores]
        return torch.stack(k_pad), torch.stack(s_pad)


class DINOv3FeatureExtractor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
            self.dino = AutoModel.from_pretrained(cfg.DINO_LOCAL_PATH, local_files_only=True)
        except:
            self.dino = AutoModel.from_pretrained("facebook/dinov2-base")
        self.patch = self.dino.config.patch_size
        self.feat_dim = self.dino.config.hidden_size

    def forward(self, img, kpts):
        with torch.no_grad():
            out = self.dino(img).last_hidden_state
        B, _, H, W = img.shape
        n_h, n_w = H // self.patch, W // self.patch
        feat = out[:, -(n_h * n_w):].transpose(1, 2).reshape(B, self.feat_dim, n_h, n_w)
        grid = kpts.clone()
        grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
        return F.grid_sample(feat, grid.unsqueeze(2), align_corners=True).squeeze(3).transpose(1, 2)


class SparseMatchingStereoModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.det = SparseKeypointDetector(cfg)
        self.ext = DINOv3FeatureExtractor(cfg)
        self.proj = torch.nn.Linear(2, cfg.FEATURE_DIM)
        layer = torch.nn.TransformerEncoderLayer(d_model=cfg.FEATURE_DIM, nhead=cfg.NUM_HEADS, batch_first=True)
        self.trans = torch.nn.TransformerEncoder(layer, num_layers=cfg.NUM_ATTENTION_LAYERS)
        self.out_proj = torch.nn.Linear(cfg.FEATURE_DIM, cfg.FEATURE_DIM)

    def forward(self, lg, rg, lrgb, rrgb, mask):
        kpl, sl = self.det(lg, mask)
        kpr, sr = self.det(rg, torch.ones_like(rg))
        descl = self.ext(lrgb, kpl)
        descr = self.ext(rrgb, kpr)
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
        return kpl, disp, scores


# --- 工具函数 ---
def load_calibration(calib_file):
    data = np.load(calib_file)
    return data['Q'], data['map1_left'], data['map2_left'], data['map1_right'], data['map2_right'], tuple(
        map(int, data['roi_left'])), tuple(map(int, data['roi_right']))


def preprocess_image(path, map1, map2, roi, mask_thresh, target_size=None):
    img = cv2.imread(path, 0)
    if img is None: return None, None, None
    img_rect = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    x, y, w, h = roi
    img_crop = img_rect[y:y + h, x:x + w]
    if target_size:
        img_crop = img_crop[:target_size[0], :target_size[1]]
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
    return np.stack(
        [homog_points[valid, 0] / w[valid], homog_points[valid, 1] / w[valid], homog_points[valid, 2] / w[valid]],
        axis=1)


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
    return (R_mat @ centered.T).T


def wave_func(z, A, k, phi, offset):
    return A * np.cos(k * z + phi) + offset


# --- [核心升级] 放宽筛选标准 ---
def fit_wave_relaxed(z, y, sigma_thresh=3.0, max_iter=3):
    # 1. 初始猜测
    k_init = 2 * np.pi / 2500
    A_init = (np.max(y) - np.min(y)) / 2
    p0 = [A_init, k_init, 0, np.mean(y)]

    mask = np.ones_like(y, dtype=bool)
    popt = p0

    # 注意：这里改用了 3.0 倍标准差 (99.7% 置信区间)
    print(f"\n--- Relaxed Sigma Clipping (Thresh={sigma_thresh}σ) ---")
    for i in range(max_iter):
        try:
            popt, _ = curve_fit(wave_func, z[mask], y[mask], p0=popt, maxfev=20000)

            y_pred = wave_func(z, *popt)
            residuals = np.abs(y - y_pred)

            current_std = np.std(residuals[mask])
            limit = sigma_thresh * current_std
            limit = max(limit, 20.0)  # 增加底线宽容度到 20mm

            new_mask = residuals < limit
            n_dropped = mask.sum() - new_mask.sum()

            print(f"Iter {i + 1}: std={current_std:.2f}mm, limit={limit:.2f}mm, dropped {n_dropped} points")

            if n_dropped == 0: break
            if new_mask.sum() < 10: break
            mask = new_mask
        except Exception as e:
            print(f"Fit error: {e}")
            break

    raw_h = abs(popt[0]) * 2
    return popt, mask, raw_h


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    model = SparseMatchingStereoModel(cfg).to(device)
    if not os.path.exists(cfg.CHECKPOINT_PATH):
        print("Checkpoint not found.");
        return
    model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location=device), strict=True)
    model.eval()

    Q, m1l, m2l, m1r, m2r, roi_l, roi_r = load_calibration(cfg.CALIB_PATH)
    l_files = sorted(glob.glob(os.path.join(cfg.LEFT_IMG_DIR, "left*.*")))
    if not l_files: print("No images."); return

    l_path = l_files[0]
    print(f"Processing Image: {os.path.basename(l_path)}")
    basename = os.path.basename(l_path)
    r_basename = "right" + basename[4:] if basename.startswith("left") else basename.replace("left", "right")
    r_path = os.path.join(cfg.RIGHT_IMG_DIR, r_basename)

    target_size = (min(roi_l[3], roi_r[3]), min(roi_l[2], roi_r[2]))
    lg, lrgb, lm = preprocess_image(l_path, m1l, m2l, roi_l, cfg.MASK_THRESHOLD, target_size)
    rg, rrgb, _ = preprocess_image(r_path, m1r, m2r, roi_r, cfg.MASK_THRESHOLD, target_size)

    if lg is None: return

    def pad14(t):
        h, w = t.shape[-2:]
        return F.pad(t, (0, (14 - w % 14) % 14, 0, (14 - h % 14) % 14))

    inputs = [pad14(t).unsqueeze(0).to(device) for t in [lg, rg, lrgb, rrgb, lm]]

    with torch.no_grad():
        kpl, disp, scores = model(*inputs)

    kpl, disp, scores = kpl[0].cpu().numpy(), disp[0].cpu().numpy(), scores[0].cpu().numpy()
    mask_valid = (scores.max(axis=1) > cfg.CONF_THRESH) & (disp > 1.0)
    pts_3d = reproject_to_3d_optimized(kpl[mask_valid], disp[mask_valid], Q, roi_l, roi_r)

    if len(pts_3d) < 10: print("Not enough points."); return

    mask_depth = (pts_3d[:, 2] > 500) & (pts_3d[:, 2] < 15000)
    pts_pca = align_point_cloud_pca(pts_3d[mask_depth])
    pts_clean = pts_pca[np.abs(pts_pca[:, 1]) < 500]

    range_x = pts_clean[:, 0].max() - pts_clean[:, 0].min()
    range_z = pts_clean[:, 2].max() - pts_clean[:, 2].min()
    if range_x > range_z:
        Z, X = pts_clean[:, 0], pts_clean[:, 2]
    else:
        Z, X = pts_clean[:, 2], pts_clean[:, 0]
    Y = pts_clean[:, 1]

    idx = np.argsort(Z)
    Z, Y, X = Z[idx], Y[idx], X[idx]

    # --- 调用宽松版拟合 ---
    popt, mask, raw_h = fit_wave_relaxed(Z, Y)

    print("\n" + "=" * 40)
    print("   FINAL ANALYSIS RESULT (High Confidence)")
    print("=" * 40)
    print(f"Measured Wave Height : {raw_h:.1f} mm")
    print(f"Valid Points Ratio   : {mask.sum()}/{len(mask)} ({mask.sum() / len(mask) * 100:.1f}%)")

    # --- 绘图 ---
    fig = plt.figure(figsize=(18, 6))

    # 1. 3D Overview
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[mask], Z[mask], Y[mask], c='lightgreen', edgecolors='green', s=25, label='Valid Foam')
    ax1.scatter(X[~mask], Z[~mask], Y[~mask], c='red', marker='x', s=20, alpha=0.2, label='Outliers')  # 降低红点透明度

    xg = np.linspace(X.min(), X.max(), 20)
    zg = np.linspace(Z.min(), Z.max(), 100)
    Xg, Zg = np.meshgrid(xg, zg)
    Yg = wave_func(Zg, *popt)
    ax1.plot_surface(Xg, Zg, Yg, color='cyan', alpha=0.15)
    ax1.set_title('3D Reconstruction (Raw Data)')
    ax1.set_xlabel('Width X');
    ax1.set_ylabel('Prop Z');
    ax1.set_zlabel('Height Y')
    ax1.set_zlim(Y.min(), Y.max())
    ax1.view_init(30, -60);
    ax1.legend()

    # 2. Side View Fit
    ax2 = fig.add_subplot(132)
    z_line = np.linspace(Z.min(), Z.max(), 500)
    ax2.scatter(Z[mask], Y[mask], c='green', alpha=0.6, s=20, label='Inliers')
    # 离群点画淡一点
    ax2.scatter(Z[~mask], Y[~mask], c='red', marker='x', alpha=0.3, s=30, label='Outliers')

    ax2.plot(z_line, wave_func(z_line, *popt), 'k-', lw=2, label='Raw Fit')
    ax2.set_title(f'Side View: Measured Height ~{raw_h:.0f}mm')
    ax2.set_xlabel('Z (mm)');
    ax2.set_ylabel('Y (mm)')
    ax2.legend(loc='upper right');
    ax2.grid(True, alpha=0.3)

    # 3. Residuals
    ax3 = fig.add_subplot(133)
    res_valid = Y[mask] - wave_func(Z[mask], *popt)
    ax3.hist(res_valid, bins=15, color='lightgreen', edgecolor='green', alpha=0.7, label='Inliers')
    ax3.set_title(f'Residuals (3-Sigma Filtered)\nMean:{np.mean(res_valid):.2f}mm Std:{np.std(res_valid):.2f}mm')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('result_final_relaxed.png', dpi=150)
    print("\nSuccess! Visualization saved to 'result_final_relaxed.png'")


if __name__ == "__main__":
    main()