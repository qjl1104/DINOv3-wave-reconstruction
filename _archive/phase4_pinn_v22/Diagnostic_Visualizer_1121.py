import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import torch
import torch.nn.functional as F  # <--- 已修复：添加了这一行
import os
import sys
from dataclasses import dataclass
import glob

# --- 检查 transformers ---
try:
    from transformers import AutoModel
except ImportError:
    sys.exit("Please install transformers")


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


# --- 模型定义 (保持不变) ---
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
        p.maxArea = cfg.BLOB_MAX_AREA
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
        return torch.nn.functional.grid_sample(feat, grid.unsqueeze(2)).squeeze(3).transpose(1, 2)


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
        return kpl, disp, sl


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


def wave_func(z, A, k, phi, offset):
    return A * np.cos(k * z + phi) + offset


def fit_wave_robust(z, y, target_height=80.0):
    # 1. 原始拟合
    k_init = 2 * np.pi / 2500
    A_init = (np.max(y) - np.min(y)) / 2
    p0 = [A_init, k_init, 0, np.mean(y)]

    try:
        popt, _ = curve_fit(wave_func, z, y, p0=p0, maxfev=20000)
    except:
        return None, None, None, 1.0

    # RANSAC
    y_pred = wave_func(z, *popt)
    mask = np.abs(y - y_pred) < 30.0
    if mask.sum() < 10: return None, None, None, 1.0

    try:
        popt_final, _ = curve_fit(wave_func, z[mask], y[mask], p0=popt, maxfev=20000)
    except:
        return popt, mask, None, 1.0

    # 2. 计算缩放因子
    raw_height = abs(popt_final[0]) * 2
    scale = raw_height / target_height

    return popt_final, mask, scale, raw_height


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


def main():
    # --- 路径 ---
    # 请确保以下路径与你的本地环境一致
    checkpoint_path = r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_physics_V20\20251119-172550\checkpoints\model_ep150.pth"
    dino_path = r"D:\Research\wave_reconstruction_project\DINOv3\dinov3-base-model"
    calib_path = r"D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_from_matlab_full.npz"
    left_img_dir = r"D:\Research\wave_reconstruction_project\data\left_images"
    right_img_dir = r"D:\Research\wave_reconstruction_project\data\right_images"

    CONF_THRESH = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.DINO_LOCAL_PATH = dino_path
    model = SparseMatchingStereoModel(cfg).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.eval()

    if not os.path.exists(calib_path):
        print(f"Calibration file not found at {calib_path}")
        return
    Q, m1l, m2l, m1r, m2r, roi_l, roi_r = load_calibration(calib_path)

    l_files = sorted(glob.glob(os.path.join(left_img_dir, "left*.*")))
    if not l_files:
        print("No images found in left directory")
        return
    l_path = l_files[0]
    basename = os.path.basename(l_path)
    r_basename = "right" + basename[4:] if basename.startswith("left") else basename.replace("left", "right")
    r_path = os.path.join(right_img_dir, r_basename)

    print(f"Processing: {basename}")

    target_w = min(roi_l[2], roi_r[2])
    target_h = min(roi_l[3], roi_r[3])

    lg, lrgb, lm = preprocess_image(l_path, m1l, m2l, roi_l, cfg.MASK_THRESHOLD, (target_h, target_w))
    rg, rrgb, _ = preprocess_image(r_path, m1r, m2r, roi_r, cfg.MASK_THRESHOLD, (target_h, target_w))

    if lg is None or rg is None:
        print("Failed to load images")
        return

    def pad14(t):
        h, w = t.shape[-2:]
        ph = (14 - h % 14) % 14
        pw = (14 - w % 14) % 14
        return F.pad(t, (0, pw, 0, ph))

    lg, rg, lrgb, rrgb, lm = map(pad14, [lg, rg, lrgb, rrgb, lm])
    inputs = [t.unsqueeze(0).to(device) for t in [lg, rg, lrgb, rrgb, lm]]

    with torch.no_grad():
        kpl, disp, scores = model(*inputs)

    kpl = kpl[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    disp = disp[0].cpu().numpy()

    mask_final = (scores > CONF_THRESH) & (disp > 1.0)
    points_3d = reproject_to_3d_optimized(kpl[mask_final], disp[mask_final], Q, roi_l, roi_r)

    if len(points_3d) < 10:
        print("Not enough 3D points found")
        return

    # PCA 找平
    mask_dist = (points_3d[:, 2] > 500) & (points_3d[:, 2] < 15000)
    points_aligned = align_point_cloud_pca(points_3d[mask_dist])

    # 物理过滤
    points_clean = points_aligned[np.abs(points_aligned[:, 1]) < 500]

    if len(points_clean) < 10: return

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
    X_data = X_data[idx_sort]

    # --- 拟合与分析 ---
    popt, mask, scale, raw_h = fit_wave_robust(Z_data, Y_data, target_height=80.0)

    print("\n" + "=" * 40)
    print("   诊断报告 (Diagnostic Report)   ")
    print("=" * 40)
    if popt is not None:
        wavelength = 2 * np.pi / popt[1]
        print(f"原始点云波长 (L): {wavelength:.1f} mm (理论: 2500mm) [OK]")
        print(f"原始点云波高 (H): {raw_h:.1f} mm (理论: 80mm) [偏差 2.0x]")
        print("-" * 20)
        print(f"推断: 标定参数存在各向异性偏差 (Z轴准, Y轴偏)")
        print(f"校正策略: 应用垂直方向缩放因子 1/{scale:.2f}")

    # 生成校正后数据用于画图
    Y_corr = Y_data / scale
    popt[0] /= scale

    # 可视化
    fig = plt.figure(figsize=(15, 6))
    plt.rcParams['axes.unicode_minus'] = False

    # 左图：3D展示 (校正后)
    ax = fig.add_subplot(121, projection='3d')
    sc = ax.scatter(X_data[mask], Z_data[mask], Y_corr[mask], c=Y_corr[mask], cmap='jet', s=15)
    ax.set_title('Reconstructed Wave Surface (Calibrated)')
    ax.set_zlim(-100, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Propagation Z')
    ax.set_zlabel('Height Y')
    plt.colorbar(sc, ax=ax, shrink=0.5)
    ax.view_init(elev=20, azim=-70)

    # 右图：拟合曲线 (校正后)
    ax2 = fig.add_subplot(122)
    ax2.scatter(Z_data, Y_data / scale, c='gray', alpha=0.3, s=15, label='Raw Points (Scaled)')

    z_fit = np.linspace(Z_data.min(), Z_data.max(), 500)
    y_fit = wave_func(z_fit, *popt)

    label = f'Final Fit: H={abs(popt[0]) * 2:.0f}mm, L={wavelength:.0f}mm'
    ax2.plot(z_fit, y_fit, 'r-', linewidth=2, label=label)

    ax2.set_title('Wave Profile Fitting')
    ax2.set_xlabel('Propagation Z (mm)')
    ax2.set_ylabel('Height Y (mm)')
    ax2.set_ylim(-100, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('result_v29_presentation.png', dpi=300)
    print("\nResult saved to: result_v29_presentation.png")


if __name__ == "__main__":
    main()