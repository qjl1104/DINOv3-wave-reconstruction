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
    # 路径配置 (请确保 calibration_paper_params.npz 在正确位置)
    # 建议使用相对路径或确保文件名正确
    DINO_LOCAL_PATH: str = "dinov3-base-model"
    CHECKPOINT_PATH: str = r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_physics_V20\20251119-172550\checkpoints\model_ep150.pth"

    # [核心修改] 使用新生成的标定文件
    CALIB_PATH: str = "calibration_paper_params.npz"

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


# --- 模型组件 (保持不变) ---
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

        # [关键修改] 极线约束过滤
        # 因为新标定非常准，这里的阈值可以设得很小 (例如 2.0 像素)
        mask_geo = (kpl[:, :, 1].unsqueeze(2) - kpr[:, :, 1].unsqueeze(1)).abs() < 2.0
        scores = scores.masked_fill(~mask_geo, -1e9)

        probs = F.softmax(scores, dim=-1)
        x_right_ex = (probs * kpr[:, :, 0].unsqueeze(1)).sum(dim=2)
        disp = kpl[:, :, 0] - x_right_ex
        return kpl, disp, scores


# --- 工具函数 ---
def load_calibration(calib_file):
    if not os.path.exists(calib_file):
        print(f"错误: 找不到标定文件 {calib_file}")
        sys.exit(1)
    data = np.load(calib_file)
    return data['Q'], data['map1_left'], data['map2_left'], data['map1_right'], data['map2_right'], tuple(
        map(int, data['roi_left'])), tuple(map(int, data['roi_right']))


def preprocess_image(path, map1, map2, roi, mask_thresh, target_size=None):
    img = cv2.imread(path, 0)
    if img is None: return None, None, None

    # [关键] 使用 float32 的 map 进行 remap，保证精度
    img_rect = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    x, y, w, h = roi
    # 注意：如果 roi 是全图，这里切片不会有问题
    # 如果新标定 roi 很小，这里可能需要调整
    if w == 0 or h == 0:  # 容错处理
        img_crop = img_rect
    else:
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
    # 注意：如果 Q 是基于 crop 后的图像，这里不需要加 roi_l
    # 我们的新标定生成的 Q 是基于【校正后全图】的，但使用了 alpha=0 裁剪
    # cv2.stereoRectify 返回的 Q 通常直接适用于 remap 后的图像
    # 如果我们在 preprocess_image 里做了 crop，那么这里的 u, v 需要加回 crop 的偏移量

    # 这里 kpts 是在 crop 后的图上的坐标
    u_L = kpts[:, 0] + roi_l[0]
    v_L = kpts[:, 1] + roi_l[1]

    # 视差 disp = x_l - x_r
    # 在校正图像中，同名点 y 坐标相同，x 坐标之差即为视差
    # 这里的 disp 已经是 x_l - x_r 了，不需要额外修正 roi_shift，除非左右 ROI x 起点不同
    # roi_l[0] - roi_r[0] 这一项通常在 Q 矩阵的 Tx 中已经隐含了，或者通过 x_l, x_r 的绝对坐标相减抵消
    # 让我们先试直接用 disp，如果 Z 轴平移了再调

    points_vec = np.stack([u_L, v_L, disp, np.ones_like(disp)], axis=1)
    homog_points = (Q @ points_vec.T).T
    w = homog_points[:, 3]

    # 避免除以 0
    valid = np.abs(w) > 1e-6
    points_3d = np.zeros_like(homog_points[:, :3])
    points_3d[valid] = homog_points[valid, :3] / w[valid, None]

    return points_3d[valid]


def wave_func(z, A, k, phi, offset):
    return A * np.cos(k * z + phi) + offset


def fit_wave_relaxed(z, y, sigma_thresh=3.0, max_iter=3):
    # 1. 初始猜测
    k_init = 2 * np.pi / 2500
    A_init = (np.max(y) - np.min(y)) / 2
    p0 = [A_init, k_init, 0, np.mean(y)]

    mask = np.ones_like(y, dtype=bool)
    popt = p0

    print(f"\n--- Relaxed Sigma Clipping (Thresh={sigma_thresh}σ) ---")
    for i in range(max_iter):
        try:
            popt, _ = curve_fit(wave_func, z[mask], y[mask], p0=popt, maxfev=20000)
            y_pred = wave_func(z, *popt)
            residuals = np.abs(y - y_pred)
            current_std = np.std(residuals[mask])
            limit = sigma_thresh * current_std
            limit = max(limit, 20.0)
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

    print(f"使用标定文件: {cfg.CALIB_PATH}")

    model = SparseMatchingStereoModel(cfg).to(device)
    if not os.path.exists(cfg.CHECKPOINT_PATH):
        print("Checkpoint not found. (跳过加载模型，仅测试流程)");
        # return # 为了测试流程，这里可以暂时注释掉 return，或者你确保路径正确
    else:
        model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location=device), strict=True)
        print("模型加载成功！")
    model.eval()

    Q, m1l, m2l, m1r, m2r, roi_l, roi_r = load_calibration(cfg.CALIB_PATH)

    # 自动搜索图片
    l_files = sorted(glob.glob(os.path.join(cfg.LEFT_IMG_DIR, "left*.*")))
    if not l_files: l_files = sorted(glob.glob(os.path.join(cfg.LEFT_IMG_DIR, "*.*")))

    if not l_files: print("No images."); return

    l_path = l_files[0]
    print(f"Processing Image: {os.path.basename(l_path)}")

    # 自动匹配右图
    basename = os.path.basename(l_path)
    if "left" in basename:
        r_basename = basename.replace("left", "right")
    elif "Left" in basename:
        r_basename = basename.replace("Left", "Right")
    else:
        r_basename = basename
    r_path = os.path.join(cfg.RIGHT_IMG_DIR, r_basename)
    if not os.path.exists(r_path):
        # 尝试找右侧第一张
        r_files = sorted(glob.glob(os.path.join(cfg.RIGHT_IMG_DIR, "*.*")))
        if r_files: r_path = r_files[0]

    target_size = (min(roi_l[3], roi_r[3]), min(roi_l[2], roi_r[2]))
    # 如果 ROI 为 0 (旧版opencv可能行为不同)，则不裁剪
    if target_size[0] == 0 or target_size[1] == 0: target_size = None

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

    # 过滤有效点
    # 因为新标定下焦距很大 (7000+)，视差 disp 也会很大
    # Z = f * B / disp => disp = f * B / Z
    # 假设 Z = 3000mm, f = 7000, B = 1400 => disp = 7000 * 1400 / 3000 ≈ 3266 像素
    # 所以 disp > 1.0 这个阈值太小了，几乎所有点都会过。但没关系。
    mask_valid = (scores.max(axis=1) > cfg.CONF_THRESH) & (disp > 10.0)  # 提高一点阈值

    pts_3d = reproject_to_3d_optimized(kpl[mask_valid], disp[mask_valid], Q, roi_l, roi_r)

    if len(pts_3d) < 10: print("Not enough points."); return

    # 空间过滤 (根据实际场景调整)
    # Z 轴通常是深度方向 (垂直于相机平面)
    # Y 轴通常是高度方向 (垂直于水面) -> 或者是 X 轴?
    # 严志勇论文中坐标系可能是: Z向前, Y向下, X向右
    # 让我们先不过滤太狠，看看原始数据

    pts_clean = pts_3d
    # mask_depth = (pts_3d[:, 2] > 1000) & (pts_3d[:, 2] < 20000)
    # pts_clean = pts_3d[mask_depth]

    # 坐标轴映射 (假设 Y 是高度)
    X = pts_clean[:, 0]
    Y = pts_clean[:, 1]
    Z = pts_clean[:, 2]  # 深度

    # 为了拟合波浪，通常波浪沿 Z (深度) 传播，或者沿 X (宽度) 传播
    # 假设波浪传播方向是 Z，高度是 Y
    # 我们对 Z 进行排序
    idx = np.argsort(Z)
    Z, Y, X = Z[idx], Y[idx], X[idx]

    # --- 调用拟合 ---
    # 注意：如果数据本身已经是波浪形状，这里会拟合得很好
    # 如果数据是乱的，这里会报错或拟合出奇怪的东西
    try:
        popt, mask, raw_h = fit_wave_relaxed(Z, Y)
    except:
        print("拟合失败，跳过拟合步骤。")
        mask = np.ones_like(Y, dtype=bool)
        popt = [0, 0, 0, 0]
        raw_h = 0

    print("\n" + "=" * 40)
    print("   FINAL ANALYSIS RESULT")
    print("=" * 40)
    print(f"Measured Wave Height : {raw_h:.1f} mm")
    print(f"Valid Points : {len(Z)}")

    # --- 绘图 ---
    fig = plt.figure(figsize=(18, 6))

    # 1. 3D Overview
    ax1 = fig.add_subplot(131, projection='3d')
    # 降采样绘图，防止太卡
    step = max(1, len(X) // 1000)
    ax1.scatter(X[::step], Z[::step], Y[::step], c=Y[::step], cmap='viridis', s=5)

    ax1.set_title('3D Reconstruction (New Calib)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (Depth)')
    ax1.set_zlabel('Y (Height)')

    # 自动调整坐标轴比例
    # ax1.set_box_aspect([1,1,0.5])

    # 2. Side View (Z-Y)
    ax2 = fig.add_subplot(132)
    ax2.scatter(Z, Y, c='blue', alpha=0.3, s=5, label='Points')
    if raw_h > 0:
        z_line = np.linspace(Z.min(), Z.max(), 500)
        ax2.plot(z_line, wave_func(z_line, *popt), 'r-', lw=2, label='Fit')
    ax2.set_title(f'Side View (Depth vs Height)')
    ax2.set_xlabel('Z (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Front View (X-Y) - 检查横向倾斜
    ax3 = fig.add_subplot(133)
    ax3.scatter(X, Y, c='green', alpha=0.3, s=5)
    ax3.set_title('Front View (Width vs Height)')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('result_final_v21.png', dpi=150)
    print("\nSuccess! Visualization saved to 'result_final_v21.png'")


if __name__ == "__main__":
    main()