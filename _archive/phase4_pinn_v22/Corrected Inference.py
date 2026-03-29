# inference_v21_fixed.py
# [V21.1 推理逻辑修正版]
# ----------------------------------------------------------------------
# 核心修复:
# 1. [坐标系取值] 修改了 h_data, x_data, y_data 的取值逻辑。
#    Z轴 (Index 2) 才是波高/深度，之前错误地取了 Y 轴。
# 2. [调试可视化] 增加了 `debug_cam_coords.png`，在 RANSAC 前
#    先画图确认相机坐标系下的波形，方便排查问题。
# ----------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from dataclasses import dataclass
from sklearn.linear_model import RANSACRegressor

try:
    from transformers import AutoModel
except ImportError:
    print("transformers not found.")
    sys.exit(1)

# --- 路径配置 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


@dataclass
class Config:
    LEFT_IMAGE_DIR: str = ""
    RIGHT_IMAGE_DIR: str = ""
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")
    DINO_LOCAL_PATH: str = "dinov3-base-model"
    MASK_THRESHOLD: int = 30
    MAX_KEYPOINTS: int = 1024
    BLOB_MIN_THRESHOLD: float = 15.0
    BLOB_MIN_AREA: float = 5.0
    BLOB_MAX_AREA: float = 2500.0
    FEATURE_DIM: int = 768
    NUM_ATTENTION_LAYERS: int = 6
    NUM_HEADS: int = 8
    MATCHING_TEMPERATURE: float = 15.0


# --- 模型定义 (保持不变) ---
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
        max_l = max([len(k) for k in kpts]) if kpts else 0
        if max_l == 0: return torch.zeros(B, 1, 2).to(img.device), torch.zeros(B, 1).to(img.device)
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
            print(f"Loading DINO online...")
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


# --- 工具函数 ---
def load_calibration(calib_file):
    calib = np.load(calib_file)
    return (calib['Q'], calib['map1_left'], calib['map2_left'],
            calib['map1_right'], calib['map2_right'],
            tuple(map(int, calib['roi_left'])), tuple(map(int, calib['roi_right'])))


def correct_Q_matrix(Q_base, roi):
    Q = Q_base.copy()
    crop_x, crop_y = roi[0], roi[1]
    Q[0, 3] += crop_x
    Q[1, 3] += crop_y
    return Q


def reproject_to_3d(keypoints, disparity, Q):
    N = len(keypoints)
    points_4d = np.zeros((N, 4))
    points_4d[:, 0] = keypoints[:, 0]
    points_4d[:, 1] = keypoints[:, 1]
    points_4d[:, 2] = disparity
    points_4d[:, 3] = 1.0
    projected = (Q @ points_4d.T).T
    X = projected[:, 0] / projected[:, 3]
    Y = projected[:, 1] / projected[:, 3]
    Z = projected[:, 2] / projected[:, 3]
    return np.stack([X, Y, Z], axis=1)


def wave_residuals_polar(params, x, y, h_obs):
    A, k_mag, theta, phi, C = params
    kx = k_mag * np.cos(theta)
    ky = k_mag * np.sin(theta)
    h_pred = A * np.cos(kx * x + ky * y + phi) + C
    return h_pred - h_obs


def ransac_auto_rectification(points_3d):
    if len(points_3d) < 10: return points_3d, None

    X_plane = points_3d[:, :2]
    z_plane = points_3d[:, 2]

    ransac = RANSACRegressor(random_state=42, residual_threshold=50.0)
    try:
        ransac.fit(X_plane, z_plane)
    except:
        return points_3d, None

    a, b = ransac.estimator_.coef_
    # 构建法向量，注意：如果平面主要是 XY 平面，法向量应接近 Z 轴
    normal = np.array([a, b, -1.0])
    normal = normal / np.linalg.norm(normal)

    if np.dot(normal, [0, 0, 1]) < 0:  # 强制朝上
        normal = -normal

    # 目标 Z 轴
    z_new = normal
    # 构造新的 X 轴 (在原 XY 平面上找一个向量垂直于 z_new)
    # 简单方法：叉乘一个非平行向量
    x_temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(x_temp, normal)) > 0.9:
        x_temp = np.array([0.0, 1.0, 0.0])

    y_new = np.cross(z_new, x_temp)
    y_new = y_new / np.linalg.norm(y_new)
    x_new = np.cross(y_new, z_new)

    R = np.vstack([x_new, y_new, z_new])

    centroid = np.mean(points_3d[ransac.inlier_mask_], axis=0)
    centered = points_3d - centroid
    rotated = (R @ centered.T).T

    return rotated, ransac.inlier_mask_


def main():
    # 请修改为你的实际模型路径
    default_model_path = r"D:\Research\wave_reconstruction_project\DINOv3\training_runs_physics_V21\checkpoints\model_ep150.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Model .pth path")
    parser.add_argument("--dino_path", type=str, default="dinov3-base-model")
    parser.add_argument("--left_img_dir", type=str, default=os.path.join(DATA_ROOT, "data", "left_images"))
    parser.add_argument("--right_img_dir", type=str, default=os.path.join(DATA_ROOT, "data", "right_images"))
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[Error] 找不到模型文件: {args.model_path}")
        # sys.exit(1) # 允许继续，方便调试代码逻辑

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.DINO_LOCAL_PATH = args.dino_path

    model = SparseMatchingStereoModel(cfg).to(device)
    try:
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load strict failed, trying non-strict... {e}")
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    Q_base, map1_l, map2_l, map1_r, map2_r, roi_l, roi_r = load_calibration(cfg.CALIBRATION_FILE)

    left_imgs = sorted(glob.glob(os.path.join(args.left_img_dir, "*.*")))
    if not left_imgs:
        print("No images found")
        return

    l_path = left_imgs[0]
    # 简单的文件名匹配
    filename = os.path.basename(l_path)
    if "left" in filename:
        r_name = filename.replace("left", "right")
    else:
        r_name = "right" + filename[4:]
    r_path = os.path.join(args.right_img_dir, r_name)

    print(f"Processing {os.path.basename(l_path)}...")

    l_raw = cv2.imread(l_path, 0)
    r_raw = cv2.imread(r_path, 0)
    l_rect = cv2.remap(l_raw, map1_l, map2_l, cv2.INTER_LINEAR)
    r_rect = cv2.remap(r_raw, map1_r, map2_r, cv2.INTER_LINEAR)

    l_crop = l_rect[roi_l[1]:roi_l[1] + roi_l[3], roi_l[0]:roi_l[0] + roi_l[2]]
    r_crop = r_rect[roi_r[1]:roi_r[1] + roi_r[3], roi_r[0]:roi_r[0] + roi_r[2]]
    th = min(l_crop.shape[0], r_crop.shape[0])
    tw = min(l_crop.shape[1], r_crop.shape[1])
    l_crop, r_crop = l_crop[:th, :tw], r_crop[:th, :tw]

    ph, pw = (14 - th % 14) % 14, (14 - tw % 14) % 14
    l_pad = cv2.copyMakeBorder(l_crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    r_pad = cv2.copyMakeBorder(r_crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    _, mask_np = cv2.threshold(l_pad, cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

    lg = torch.from_numpy(l_pad).float().unsqueeze(0).unsqueeze(0) / 255.0
    rg = torch.from_numpy(r_pad).float().unsqueeze(0).unsqueeze(0) / 255.0
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    lrgb = torch.from_numpy(cv2.cvtColor(l_pad, cv2.COLOR_GRAY2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    rrgb = torch.from_numpy(cv2.cvtColor(r_pad, cv2.COLOR_GRAY2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        out = model(lg.to(device), rg.to(device), lrgb.to(device), rrgb.to(device), mask.to(device))

    kp = out['keypoints_left'][0].cpu().numpy()
    disp = out['disparity'][0].cpu().numpy()
    score = out['scores_left'][0].cpu().numpy()

    valid = (score > 0.1) & (disp > 0.1) & (kp[:, 0] < tw) & (kp[:, 1] < th)
    kp_v = kp[valid]
    disp_v = disp[valid]

    roi_disp_offset = roi_l[0] - roi_r[0]
    disp_real = disp_v + roi_disp_offset

    Q_corr = correct_Q_matrix(Q_base, roi_l)
    pts_cam = reproject_to_3d(kp_v, disp_real, Q_corr)

    valid_mask = (pts_cam[:, 2] > 500) & (pts_cam[:, 2] < 20000)
    pts_clean = pts_cam[valid_mask]

    if len(pts_clean) < 50:
        print("Not enough points.")
        return

    # --- [新增] 调试绘图: 检查相机坐标系下的点云 ---
    print("Drawing Debug Camera Coordinates...")
    plt.figure(figsize=(10, 5))
    # 通常相机坐标系下：Y轴向下，Z轴向前(深度)
    # 波浪如果主要在深度方向变化，看 Y-Z 图或者 X-Z 图
    plt.scatter(pts_clean[:, 1], pts_clean[:, 2], s=2, alpha=0.5)
    plt.xlabel('Y (Camera)')
    plt.ylabel('Z (Depth)')
    plt.title('Raw Camera Coordinates (Check for Wave Shape)')
    plt.grid(True)
    plt.savefig('debug_cam_coords.png')
    print("Saved debug_cam_coords.png. 如果这里看不到波形，说明DINO匹配失败。")

    print("RANSAC Auto-Rectification...")
    pts_world, inlier_mask = ransac_auto_rectification(pts_clean)

    # RANSAC 校正后，新的 Z 轴应该是垂直于水面的（即波高方向）
    # 过滤离群点
    if inlier_mask is not None:
        pts_world_clean = pts_world[inlier_mask]
    else:
        pts_world_clean = pts_world

    # [核心修正] 坐标轴取值
    # 在 RANSAC 旋转后的坐标系中：
    # Z轴 (Index 2) = 垂直于平均水面的高度（波高）
    # X轴 (Index 0) = 水平轴 1
    # Y轴 (Index 1) = 水平轴 2

    h_data = pts_world_clean[:, 2]  # 修改为 Z 轴作为波高
    x_data = pts_world_clean[:, 0]
    y_data = pts_world_clean[:, 1]

    # 简单的离群值过滤 (高度)
    h_mask = np.abs(h_data) < 300
    h_data = h_data[h_mask]
    x_data = x_data[h_mask]
    y_data = y_data[h_mask]

    print(f"Cleaned points: {len(h_data)}")
    print("Fitting 2D Wave Surface (Polar Constraints)...")

    try:
        A_guess = 40.0
        k_guess = 2 * np.pi / 2500
        k_min = 2 * np.pi / 3500
        k_max = 2 * np.pi / 1500

        lower_bounds = [0, k_min, -np.pi, -np.pi, -100]
        upper_bounds = [200, k_max, np.pi, np.pi, 100]
        p0 = [A_guess, k_guess, 0, 0, 0]

        res = least_squares(
            wave_residuals_polar,
            p0,
            bounds=(lower_bounds, upper_bounds),
            args=(x_data, y_data, h_data),
            loss='soft_l1',
            f_scale=20.0,
            max_nfev=20000
        )

        params = res.x
        A, k_mag, theta, phi, C = params

        real_height = abs(A * 2)
        real_wavelength = 2 * np.pi / k_mag

        print("\n" + "=" * 50)
        print(f" 【V21.1 最终结果】")
        print(f"  预测波高 (Wave Height): {real_height:.2f} mm")
        print(f"  预测波长 (Wavelength) : {real_wavelength:.2f} mm")
        print("=" * 50 + "\n")

        # 绘图: 投影到传播方向
        kx = k_mag * np.cos(theta)
        ky = k_mag * np.sin(theta)
        prop_dist = (kx * x_data + ky * y_data) / k_mag

        plt.figure(figsize=(12, 6))
        plt.scatter(prop_dist, h_data, alpha=0.5, s=10, label='Cleaned Points (Z-axis)')
        x_range = np.linspace(prop_dist.min(), prop_dist.max(), 500)
        y_fit = A * np.cos(k_mag * x_range + phi) + C
        plt.plot(x_range, y_fit, 'r', lw=2, label=f'Fit (L={real_wavelength:.0f}mm)')
        plt.xlabel('Projected Propagation Distance [mm]')
        plt.ylabel('Wave Height (Z) [mm]')
        plt.title(f'Physics-Informed Reconstruction\nHeight: {real_height:.1f}mm | Wavelength: {real_wavelength:.0f}mm')
        plt.ylim(-200, 200)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('result_v21_fixed.png')
        print("结果图已保存至 result_v21_fixed.png")

    except Exception as e:
        print(f"Fitting failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()