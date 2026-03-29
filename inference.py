"""
DINOv3 Wave Reconstruction - Single-Frame Inference
=====================================================
Loads a trained model, processes one stereo pair, and outputs:
  - 3D point cloud visualization
  - Side view with wave fitting
  - Original image
"""

import os
import sys
import glob
import argparse
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# --- Shared modules ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, PROJECT_ROOT
from models import SparseMatchingStereoModel


def load_calibration(calib_file):
    """Load calibration data (Q matrix + rectification maps)."""
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file {calib_file} not found.")
        sys.exit(1)
    data = np.load(calib_file)
    return data['Q'], data['map1_left'], data['map2_left'], data['map1_right'], data['map2_right']


def preprocess_image(path, map1, map2, mask_thresh):
    """Load, rectify, and convert a single image to tensors."""
    img = cv2.imread(path, 0)
    if img is None:
        return None, None, None

    img_rect = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    _, mask = cv2.threshold(img_rect, mask_thresh, 255, cv2.THRESH_BINARY)

    g_tensor = torch.from_numpy(img_rect).float().unsqueeze(0) / 255.0
    m_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
    rgb = cv2.cvtColor(img_rect, cv2.COLOR_GRAY2RGB)
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return g_tensor, rgb_tensor, m_tensor


def pad_to_14(t):
    """Pad tensor spatial dims to be divisible by 14."""
    h, w = t.shape[-2:]
    return F.pad(t, (0, (14 - w % 14) % 14, 0, (14 - h % 14) % 14))


def reproject_to_3d(kpts, disp, Q):
    """Reproject 2D keypoints + disparity to 3D using Q matrix."""
    u_L = kpts[:, 0]
    v_L = kpts[:, 1]
    points_vec = np.stack([u_L, v_L, disp, np.ones_like(disp)], axis=1)
    homog_points = (Q @ points_vec.T).T
    w = homog_points[:, 3]
    valid = np.abs(w) > 1e-6
    points_3d = np.zeros_like(homog_points[:, :3])
    points_3d[valid] = homog_points[valid, :3] / w[valid, None]
    return points_3d[valid]


def wave_func(z, A, k, phi, offset):
    """Cosine wave function for fitting."""
    return A * np.cos(k * z + phi) + offset


def fit_wave_relaxed(z, y):
    """Fit a 1D cosine wave to the data."""
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
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image_index", type=int, default=0,
                        help="Index of image pair to process")
    parser.add_argument("--output", type=str, default="inference_result.png",
                        help="Output figure path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Override checkpoint if provided
    if args.checkpoint:
        cfg.CHECKPOINT_PATH = args.checkpoint

    # Load calibration
    print(f"Loading Calibration: {cfg.CALIBRATION_FILE}")
    Q, m1l, m2l, m1r, m2r = load_calibration(cfg.CALIBRATION_FILE)

    # Build and load model
    model = SparseMatchingStereoModel(cfg).to(device)
    if cfg.CHECKPOINT_PATH and os.path.exists(cfg.CHECKPOINT_PATH):
        checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        print(f"Model loaded: {cfg.CHECKPOINT_PATH}")
    else:
        print("[Warning] No valid checkpoint. Running with initialized weights.")
    model.eval()

    # Find images
    l_files = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "left*.*")))
    if not l_files:
        l_files = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
    if not l_files:
        print("No images found.")
        return

    idx = min(args.image_index, len(l_files) - 1)
    l_path = l_files[idx]
    print(f"Processing: {os.path.basename(l_path)}")

    # Infer right image path
    basename = os.path.basename(l_path)
    if "left" in basename:
        r_basename = basename.replace("left", "right")
    elif "Left" in basename:
        r_basename = basename.replace("Left", "Right")
    else:
        r_basename = basename
    r_path = os.path.join(cfg.RIGHT_IMAGE_DIR, r_basename)

    # Preprocess
    lg, lrgb, lm = preprocess_image(l_path, m1l, m2l, cfg.MASK_THRESHOLD)
    rg, rrgb, _ = preprocess_image(r_path, m1r, m2r, cfg.MASK_THRESHOLD)
    if lg is None or rg is None:
        print("Failed to load images.")
        return

    inputs = [pad_to_14(t).unsqueeze(0).to(device) for t in [lg, rg, lrgb, rrgb, lm]]

    # Inference (with epipolar mask)
    with torch.no_grad():
        out = model(*inputs, apply_epipolar_mask=True)

    kpl = out['keypoints_left'][0].cpu().numpy()
    disp = out['disparity'][0].cpu().numpy()
    scores = out['match_scores'][0].cpu().numpy()

    # Filter by confidence and minimum disparity
    mask_valid = (scores.max(axis=1) > cfg.CONF_THRESH) & (disp > 50.0)
    pts_3d = reproject_to_3d(kpl[mask_valid], disp[mask_valid], Q)

    if len(pts_3d) < 10:
        print("Not enough valid points.")
        return

    X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]

    # Filter depth outliers
    mask_z = (Z > 1000) & (Z < 20000)
    X, Y, Z = X[mask_z], Y[mask_z], Z[mask_z]

    if len(X) < 10:
        print("Not enough points after depth filtering.")
        return

    # Visualization
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X, Z, Y, c=Y, cmap='viridis', s=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (Depth)')
    ax1.set_zlabel('Y (Height)')
    ax1.set_title("3D Point Cloud")

    ax2 = fig.add_subplot(132)
    ax2.scatter(Z, Y, s=2, alpha=0.5)
    ax2.set_xlabel('Depth Z (mm)')
    ax2.set_ylabel('Height Y (mm)')
    ax2.set_title("Side View")

    idx_sort = np.argsort(Z)
    popt, h_wave = fit_wave_relaxed(Z[idx_sort], Y[idx_sort])
    if h_wave > 0:
        ax2.plot(Z[idx_sort], wave_func(Z[idx_sort], *popt), 'r-', label=f'H={h_wave:.1f}mm')
        ax2.legend()

    ax3 = fig.add_subplot(133)
    orig_img = cv2.imread(l_path)
    if orig_img is not None:
        ax3.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    ax3.set_title("Original Image")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()