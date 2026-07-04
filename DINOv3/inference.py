"""
DINOv3 Wave Reconstruction - Single-Frame Inference
=====================================================
Uses the trained CorrMatchingStereoModel for end-to-end inference:
  - Blob keypoint detection + DINOv3 feature extraction
  - 1D Correlation Volume + Soft-argmax disparity regression
  - 3D reconstruction + RANSAC plane fitting + visualization

Outputs:
  - 3D wave surface point cloud (plane-subtracted wave height)
  - Side view with wave fitting
  - Top view (bird's eye)
  - Original image with keypoints overlay
"""

import os
import sys
import glob
import argparse
import warnings

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, PROJECT_ROOT
from models import CorrMatchingStereoModel
from utils import reproject_to_3d, infer_right_path, load_model_checkpoint


def load_calibration(calib_file):
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file {calib_file} not found.")
        sys.exit(1)
    data = np.load(calib_file)
    return data['Q'], data['map1_left'], data['map2_left'], data['map1_right'], data['map2_right']


def fit_plane_ransac(pts_3d, residual_thresh=30.0, min_samples=3):
    from sklearn.linear_model import RANSACRegressor
    X_feat = pts_3d[:, [0, 2]]
    Y_target = pts_3d[:, 1]
    ransac = RANSACRegressor(residual_threshold=residual_thresh, min_samples=min_samples, random_state=42)
    ransac.fit(X_feat, Y_target)
    Y_plane = ransac.predict(X_feat)
    inlier_mask = ransac.inlier_mask_
    return Y_plane, inlier_mask


def wave_func(z, A, k, phi, offset):
    return A * np.cos(k * z + phi) + offset


def fit_wave_relaxed(z, y):
    k_init = 2 * np.pi / 2500
    A_init = (np.max(y) - np.min(y)) / 2
    p0 = [A_init, k_init, 0, np.mean(y)]
    try:
        popt, _ = curve_fit(wave_func, z, y, p0=p0, maxfev=10000)
        return popt, abs(popt[0]) * 2
    except Exception:
        return [0, 0, 0, 0], 0


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Single-Frame Inference")
    parser.add_argument("--image_index", type=int, default=0,
                        help="Index of image pair to process")
    parser.add_argument("--output", type=str, default="inference_result.png",
                        help="Output figure path")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to model checkpoint (uses config default if empty)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    args = parser.parse_args()

    cfg = Config()

    checkpoint_path = args.checkpoint or cfg.CHECKPOINT_PATH
    if not checkpoint_path:
        print("Error: No checkpoint specified. Use --checkpoint or set CHECKPOINT_PATH in config.")
        sys.exit(1)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading Calibration: {cfg.CALIBRATION_FILE}")
    Q, m1l, m2l, m1r, m2r = load_calibration(cfg.CALIBRATION_FILE)

    l_files = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "left*.*")))
    if not l_files:
        l_files = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
    if not l_files:
        print("No images found.")
        return

    idx = min(args.image_index, len(l_files) - 1)
    l_path = l_files[idx]
    r_path = infer_right_path(l_path, cfg.RIGHT_IMAGE_DIR)
    print(f"Processing: {os.path.basename(l_path)}")

    l_raw = cv2.imread(l_path, 0)
    r_raw = cv2.imread(r_path, 0)
    l_rgb = cv2.imread(l_path)
    r_rgb = cv2.imread(r_path)
    if l_raw is None or r_raw is None:
        print("Failed to load images.")
        return

    l_rect = cv2.remap(l_raw, m1l, m2l, cv2.INTER_LINEAR)
    r_rect = cv2.remap(r_raw, m1r, m2r, cv2.INTER_LINEAR)
    l_rgb_rect = cv2.remap(l_rgb, m1l, m2l, cv2.INTER_LINEAR)
    r_rgb_rect = cv2.remap(r_rgb, m1r, m2r, cv2.INTER_LINEAR)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {checkpoint_path}...")
    model = CorrMatchingStereoModel(cfg).to(device)
    load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    lg = torch.from_numpy(l_rect).float().unsqueeze(0).unsqueeze(0) / 255.0
    rg = torch.from_numpy(r_rect).float().unsqueeze(0).unsqueeze(0) / 255.0
    lrgb = torch.from_numpy(l_rgb_rect).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    rrgb = torch.from_numpy(r_rgb_rect).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    mask = torch.ones(1, 1, l_rect.shape[0], l_rect.shape[1])

    lg = lg.to(device)
    rg = rg.to(device)
    lrgb = lrgb.to(device)
    rrgb = rrgb.to(device)
    mask = mask.to(device)

    print("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            out = model(lg, rg, lrgb, rrgb, mask)

    kpl = out['keypoints_left'][0].cpu().numpy()
    scores = out['scores_left'][0].cpu().numpy()
    disparity = out['disparity'][0].cpu().numpy()

    valid = (scores > 0) & (disparity > 10.0)
    kp_valid = kpl[valid]
    disp_valid = disparity[valid]
    scores_valid = scores[valid]
    print(f"Valid matches: {valid.sum()}/{len(kpl)}")

    if len(kp_valid) < 10:
        print("Not enough valid matches.")
        return

    pts_3d = reproject_to_3d(kp_valid, disp_valid, Q)
    if len(pts_3d) < 10:
        print("Not enough valid 3D points.")
        return

    X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
    mask_z = (Z > 2000) & (Z < 15000)
    pts_filt = pts_3d[mask_z]
    kp_filt = kp_valid[mask_z]
    scores_filt = scores_valid[mask_z]
    X, Y, Z = pts_filt[:, 0], pts_filt[:, 1], pts_filt[:, 2]

    if len(X) < 10:
        print("Not enough points after depth filtering.")
        return

    z_median = np.median(Z)
    z_iqr = np.percentile(Z, 75) - np.percentile(Z, 25)
    z_mask = np.abs(Z - z_median) < 2.0 * z_iqr
    pts_filt = pts_filt[z_mask]
    kp_filt = kp_filt[z_mask]
    scores_filt = scores_filt[z_mask]
    X, Y, Z = pts_filt[:, 0], pts_filt[:, 1], pts_filt[:, 2]

    if len(X) < 10:
        print("Not enough points after IQR filtering.")
        return

    Y_plane, inlier_mask = fit_plane_ransac(pts_filt, residual_thresh=25.0)
    wave_height = Y - Y_plane

    X_in = X[inlier_mask]
    Z_in = Z[inlier_mask]
    wh_in = wave_height[inlier_mask]
    kp_in = kp_filt[inlier_mask]
    scores_in = scores_filt[inlier_mask]

    wh_std = np.std(wh_in)
    wh_range = np.max(wh_in) - np.min(wh_in)

    print(f"--- Results ---")
    print(f"Total 3D points: {len(X)}")
    print(f"Inlier points (RANSAC): {inlier_mask.sum()}")
    print(f"Depth range: Z=[{Z_in.min():.0f}, {Z_in.max():.0f}] mm")
    print(f"Wave height std: {wh_std:.2f} mm")
    print(f"Wave height range: {wh_range:.2f} mm")
    print(f"Wave height: [{wh_in.min():.1f}, {wh_in.max():.1f}] mm")
    print(f"Mean score: {scores_in.mean():.3f}")

    fig = plt.figure(figsize=(20, 5))

    ax1 = fig.add_subplot(141, projection='3d')
    sc1 = ax1.scatter(X_in, Z_in, wh_in, c=wh_in, cmap='coolwarm', s=3,
                      vmin=-wh_std * 3, vmax=wh_std * 3)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_zlabel('Wave Height (mm)')
    ax1.set_title(f"Wave Surface (n={len(X_in)})")
    plt.colorbar(sc1, ax=ax1, shrink=0.5, label='Height (mm)')

    ax2 = fig.add_subplot(142)
    ax2.scatter(Z_in, wh_in, s=2, alpha=0.5, c='steelblue')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Depth Z (mm)')
    ax2.set_ylabel('Wave Height (mm)')
    ax2.set_title(f"Side View (std={wh_std:.1f}mm)")
    ax2.set_ylim(-wh_std * 4, wh_std * 4)

    idx_sort = np.argsort(Z_in)
    popt, h_wave = fit_wave_relaxed(Z_in[idx_sort], wh_in[idx_sort])
    if h_wave > 0.5:
        z_fit = np.linspace(Z_in.min(), Z_in.max(), 500)
        ax2.plot(z_fit, wave_func(z_fit, *popt), 'r-', linewidth=2,
                 label=f'H={h_wave:.1f}mm, L={2 * np.pi / abs(popt[1]):.0f}mm')
        ax2.legend()

    ax3 = fig.add_subplot(143)
    sc3 = ax3.scatter(X_in, Z_in, c=wh_in, cmap='coolwarm', s=3,
                      vmin=-wh_std * 3, vmax=wh_std * 3)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title("Top View (Bird's Eye)")
    ax3.set_aspect('equal')
    plt.colorbar(sc3, ax=ax3, shrink=0.8, label='Height (mm)')

    ax4 = fig.add_subplot(144)
    orig_img = cv2.imread(l_path)
    if orig_img is not None:
        ax4.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    ax4.scatter(kp_in[:, 0], kp_in[:, 1], c=wh_in, cmap='coolwarm', s=2,
                vmin=-wh_std * 3, vmax=wh_std * 3, alpha=0.7)
    ax4.set_title("Keypoints on Image")
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()