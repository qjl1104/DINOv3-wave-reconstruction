"""
DINOv3 Wave Reconstruction - Temporal Inference
=================================================
Processes a sequence of stereo image pairs to:
  1. Build global water surface reference plane (Phase 1)
  2. Compute per-frame wave height & wavelength (Phase 2)
  3. Perform statistical & frequency-domain analysis (Phase 3)

Uses the same calibration file and full-image mode as training.
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

# --- Shared modules ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, PROJECT_ROOT, DATA_ROOT
from models import SparseMatchingStereoModel


# ============================================================
# Calibration & Preprocessing (Full-Image Mode, matching V24)
# ============================================================

def load_calibration(calib_file):
    """Load calibration: Q matrix + rectification maps."""
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file {calib_file} not found.")
        sys.exit(1)
    calib = np.load(calib_file)
    return {
        'Q': calib['Q'].astype(np.float32),
        'map1_left': calib['map1_left'],
        'map2_left': calib['map2_left'],
        'map1_right': calib['map1_right'],
        'map2_right': calib['map2_right'],
    }


def preprocess_stereo_pair(l_path, r_path, calib, mask_thresh, device):
    """
    Load, rectify, pad, and tensorize a stereo pair.
    Returns model-ready tensors or None if loading fails.
    """
    l_raw = cv2.imread(l_path, 0)
    r_raw = cv2.imread(r_path, 0)
    if l_raw is None or r_raw is None:
        return None

    l_rect = cv2.remap(l_raw, calib['map1_left'], calib['map2_left'], cv2.INTER_LINEAR)
    r_rect = cv2.remap(r_raw, calib['map1_right'], calib['map2_right'], cv2.INTER_LINEAR)

    # Pad to multiple of 14
    h, w = l_rect.shape
    ph = (14 - h % 14) % 14
    pw = (14 - w % 14) % 14
    if ph > 0 or pw > 0:
        l_rect = cv2.copyMakeBorder(l_rect, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        r_rect = cv2.copyMakeBorder(r_rect, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)

    _, mask_np = cv2.threshold(l_rect, mask_thresh, 255, cv2.THRESH_BINARY)

    # Build tensors
    lg = torch.from_numpy(l_rect).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    rg = torch.from_numpy(r_rect).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    lrgb = torch.from_numpy(
        cv2.cvtColor(l_rect, cv2.COLOR_GRAY2RGB).transpose(2, 0, 1)
    ).float().unsqueeze(0).to(device) / 255.0
    rrgb = torch.from_numpy(
        cv2.cvtColor(r_rect, cv2.COLOR_GRAY2RGB).transpose(2, 0, 1)
    ).float().unsqueeze(0).to(device) / 255.0

    return lg, rg, lrgb, rrgb, mask


def reproject_to_3d(keypoints, disparity, Q):
    """Reproject 2D keypoints + disparity to 3D using Q matrix."""
    N = len(keypoints)
    if N == 0:
        return np.zeros((0, 3))
    points_4d = np.zeros((N, 4))
    points_4d[:, 0] = keypoints[:, 0]
    points_4d[:, 1] = keypoints[:, 1]
    points_4d[:, 2] = disparity
    points_4d[:, 3] = 1.0
    projected = (Q @ points_4d.T).T
    w = projected[:, 3]
    valid = np.abs(w) > 1e-6
    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    X[valid] = projected[valid, 0] / w[valid]
    Y[valid] = projected[valid, 1] / w[valid]
    Z[valid] = projected[valid, 2] / w[valid]
    pts = np.stack([X, Y, Z], axis=1)
    return pts[valid]


# ============================================================
# Wave fitting & Global Rotation
# ============================================================

def wave_residuals_polar(params, x, y, h_obs):
    """Residual function for 2D wave fitting."""
    A, k_mag, theta, phi, C = params
    kx = k_mag * np.cos(theta)
    ky = k_mag * np.sin(theta)
    return A * np.cos(kx * x + ky * y + phi) + C - h_obs


def compute_global_rotation(points_batch):
    """Compute rotation matrix to align water surface with XZ plane using RANSAC."""
    print(f"[Global] Computing rotation from {len(points_batch)} points...")
    if len(points_batch) < 50:
        return None

    X_plane = points_batch[:, :2]
    z_plane = points_batch[:, 2]

    ransac = RANSACRegressor(random_state=42, residual_threshold=50.0)
    try:
        ransac.fit(X_plane, z_plane)
    except Exception:
        return None

    a, b = ransac.estimator_.coef_
    normal = np.array([a, b, -1.0])
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, [0, 1, 0]) > 0:
        normal = -normal

    y_new = normal
    x_temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(x_temp, normal)) > 0.9:
        x_temp = np.array([0.0, 0.0, 1.0])
    x_new = x_temp - np.dot(x_temp, normal) * normal
    x_new = x_new / np.linalg.norm(x_new)
    z_new = np.cross(x_new, y_new)
    R = np.vstack([x_new, y_new, z_new])
    centroid = np.mean(points_batch[ransac.inlier_mask_], axis=0)
    return R, centroid


# ============================================================
# Frame Processing
# ============================================================

def process_frame(l_path, r_path, model, device, cfg, calib,
                  R_global=None, center_global=None):
    """
    Process one stereo pair: preprocess → model → 3D reproject → optional rotation.

    Returns:
        numpy array of 3D points (possibly rotated), or None
    """
    Q = calib['Q']

    # Preprocess
    tensors = preprocess_stereo_pair(l_path, r_path, calib, cfg.MASK_THRESHOLD, device)
    if tensors is None:
        return None
    lg, rg, lrgb, rrgb, mask = tensors

    # Model inference (with epipolar mask)
    with torch.no_grad():
        out = model(lg, rg, lrgb, rrgb, mask, apply_epipolar_mask=True)

    kp = out['keypoints_left'][0].cpu().numpy()
    disp = out['disparity'][0].cpu().numpy()
    score = out['scores_left'][0].cpu().numpy()

    # Get original (unpadded) image dimensions for validity check
    h_orig, w_orig = calib['map1_left'].shape[:2]

    # Filter valid keypoints
    valid = (score > 0.1) & (disp > 0.1) & (kp[:, 0] < w_orig) & (kp[:, 1] < h_orig)
    kp_v = kp[valid]
    disp_v = disp[valid]

    if len(kp_v) < 10:
        return None

    # 3D reprojection (full-image coordinates, no ROI correction needed)
    pts_cam = reproject_to_3d(kp_v, disp_v, Q)

    # Depth filter
    valid_mask = (pts_cam[:, 2] > 500) & (pts_cam[:, 2] < 20000)
    pts_clean = pts_cam[valid_mask]

    if R_global is None:
        return pts_clean

    # Apply global rotation
    centered = pts_clean - center_global
    pts_world = (R_global @ centered.T).T
    h_mask = np.abs(pts_world[:, 1]) < 400
    pts_world_clean = pts_world[h_mask]

    return pts_world_clean


# ============================================================
# Physics Analysis
# ============================================================

def analyze_physics_properties(df, fps=50):
    """Statistical and frequency-domain analysis of wave measurements."""
    print("\n--- 正在进行物理/频域分析 ---")

    # Time series (smoothed)
    h_smooth = df['height_smooth'].ffill().values
    mean_h = np.mean(h_smooth)
    std_h = np.std(h_smooth)

    # Wavelength statistics
    wavelength_series = df['wavelength'].ffill().values
    w_valid = wavelength_series[(wavelength_series > 1000) & (wavelength_series < 5000)]
    mean_lambda = np.mean(w_valid) if len(w_valid) > 0 else 0

    # Theoretical frequency from dispersion relation
    g = 9810  # mm/s^2
    if mean_lambda > 0:
        f_theoretical = np.sqrt(g / (2 * np.pi * mean_lambda))
        T_theoretical = 1.0 / f_theoretical
    else:
        f_theoretical = 0
        T_theoretical = 0

    print(f"分析帧数: {len(df)}")
    print(f"统计有效波高 (Mean Smoothed Height): {mean_h:.2f} mm (Std: {std_h:.2f})")
    print(f"统计平均波长 (Mean Wavelength): {mean_lambda:.2f} mm")
    print(f"推算波浪频率 (Derived Frequency): {f_theoretical:.2f} Hz (周期: {T_theoretical:.2f} s)")
    print("------------------------------------------------")

    # Plots
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(h_smooth[np.isfinite(h_smooth)], bins=30, color='blue', alpha=0.7, label='Height Dist.')
    plt.axvline(x=mean_h, color='red', linestyle='--', label=f'Mean: {mean_h:.1f}')
    plt.axvline(x=80, color='green', linestyle='-', linewidth=2, label='Target: 80')
    plt.title('Wave Height Distribution')
    plt.xlabel('Height [mm]')
    plt.legend()

    plt.subplot(1, 2, 2)
    if len(w_valid) > 0:
        plt.hist(w_valid, bins=30, color='green', alpha=0.7, label='Wavelength Dist.')
        plt.axvline(x=mean_lambda, color='red', linestyle='--', label=f'Mean: {mean_lambda:.0f}')
    plt.axvline(x=2500, color='blue', linestyle='-', linewidth=2, label='Target: 2500')
    plt.title('Wavelength Distribution')
    plt.xlabel('Wavelength [mm]')
    plt.legend()

    plt.tight_layout()
    plt.savefig('wave_statistics.png')
    print("Statistics plot saved to wave_statistics.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DINOv3 Temporal Wave Analysis")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--left_img_dir", type=str,
                        default=os.path.join(DATA_ROOT, "data", "left_images"))
    parser.add_argument("--right_img_dir", type=str,
                        default=os.path.join(DATA_ROOT, "data", "right_images"))
    parser.add_argument("--limit", type=int, default=999999,
                        help="Max number of frames to process")
    parser.add_argument("--output_csv", type=str, default="wave_data_temporal.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Load model
    model = SparseMatchingStereoModel(cfg).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"Model loaded: {args.model_path}")

    # Load calibration
    calib = load_calibration(cfg.CALIBRATION_FILE)

    # Find images
    left_imgs = sorted(glob.glob(os.path.join(args.left_img_dir, "left*.*")))[:args.limit]
    if not left_imgs:
        left_imgs = sorted(glob.glob(os.path.join(args.left_img_dir, "*.*")))[:args.limit]
    if not left_imgs:
        print("No images found.")
        return

    print(f"Found {len(left_imgs)} images. Starting Temporal Analysis...")

    # --- Phase 1: Calibrate Global Water Surface ---
    sample_indices = np.linspace(0, len(left_imgs) - 1, min(20, len(left_imgs))).astype(int)
    global_pts_buffer = []
    print("Phase 1: Calibrating Global Water Surface...")
    for idx in tqdm(sample_indices):
        l_path = left_imgs[idx]
        basename = os.path.basename(l_path)
        if "left" in basename:
            r_name = basename.replace("left", "right")
        elif "Left" in basename:
            r_name = basename.replace("Left", "Right")
        else:
            r_name = basename
        r_path = os.path.join(args.right_img_dir, r_name)
        pts = process_frame(l_path, r_path, model, device, cfg, calib)
        if pts is not None and len(pts) > 0:
            global_pts_buffer.append(pts)

    if not global_pts_buffer:
        print("Phase 1 failed: no valid points.")
        return
    all_sample_pts = np.concatenate(global_pts_buffer, axis=0)
    rotation_result = compute_global_rotation(all_sample_pts)
    if rotation_result is None:
        print("Phase 1 failed: could not compute rotation.")
        return
    R_global, center_global = rotation_result

    # --- Phase 2: Batch Inference ---
    results = []
    print("Phase 2: Batch Inference...")
    for i, l_path in enumerate(tqdm(left_imgs)):
        basename = os.path.basename(l_path)
        if "left" in basename:
            r_name = basename.replace("left", "right")
        elif "Left" in basename:
            r_name = basename.replace("Left", "Right")
        else:
            r_name = basename
        r_path = os.path.join(args.right_img_dir, r_name)

        pts_world = process_frame(l_path, r_path, model, device, cfg, calib,
                                  R_global, center_global)

        if pts_world is None or len(pts_world) < 20:
            results.append({'frame': i, 'height': np.nan, 'wavelength': np.nan})
            continue

        h_data = pts_world[:, 1]
        x_data = pts_world[:, 0]
        y_data = pts_world[:, 2]

        try:
            k_min, k_max = 2 * np.pi / 3500, 2 * np.pi / 1500
            lower = [0, k_min, -np.pi, -np.pi, -100]
            upper = [300, k_max, np.pi, np.pi, 100]
            p0 = [40.0, 2 * np.pi / 2500, 0, 0, 0]
            res = least_squares(wave_residuals_polar, p0, bounds=(lower, upper),
                                args=(x_data, y_data, h_data), loss='soft_l1', f_scale=20.0)
            A, k_mag, _, _, _ = res.x
            results.append({'frame': i, 'height': abs(A * 2), 'wavelength': 2 * np.pi / k_mag})
        except Exception:
            results.append({'frame': i, 'height': np.nan, 'wavelength': np.nan})

    # --- Phase 3: Statistical Analysis ---
    df = pd.DataFrame(results)
    df['height_smooth'] = df['height'].rolling(window=5, center=True).mean()
    df['wavelength_smooth'] = df['wavelength'].rolling(window=5, center=True).mean()
    df.to_csv(args.output_csv, index=False)
    print(f"Data saved to {args.output_csv}")

    # Phase 4: Physics analysis
    analyze_physics_properties(df, fps=cfg.FPS)


if __name__ == "__main__":
    main()