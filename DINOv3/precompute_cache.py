"""
DINOv3 Wave Reconstruction - Dense Feature Map Pre-computation
================================================================
Caches DINOv3 dense feature maps + blob keypoints for all stereo pairs.

Cache format (v4 - dense):
  feat_left:  [C, Hf, Wf] fp16   dense DINOv3 feature map
  feat_right: [C, Hf, Wf] fp16
  keypoints_left:  [N, 2] float32
  scores_left:     [N] float32
  keypoints_right: [M, 2] float32
  scores_right:    [M] float32
  left_gray:  [H, W] uint8
  right_gray: [H, W] uint8
  mask:       [H, W] uint8

Usage:
    python precompute_cache.py            # skip existing
    python precompute_cache.py --force    # regenerate all
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, check_path

try:
    from transformers import AutoModel
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)


def pad_to_patch(t, patch_size):
    h, w = t.shape[-2:]
    padh = (patch_size - h % patch_size) % patch_size
    padw = (patch_size - w % patch_size) % patch_size
    if padh > 0 or padw > 0:
        return F.pad(t, (0, padw, 0, padh)), (padh, padw)
    return t, (0, 0)


def detect_blobs(img_np, cfg):
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor = False
    p.minThreshold = cfg.BLOB_MIN_THRESHOLD
    p.maxThreshold = 255
    p.filterByArea = True
    p.minArea = cfg.BLOB_MIN_AREA
    p.maxArea = cfg.BLOB_MAX_AREA
    det = cv2.SimpleBlobDetector_create(p)
    kps = det.detect(img_np)
    if not kps:
        return np.zeros((1, 2), dtype=np.float32), np.zeros(1, dtype=np.float32)
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    sizes = np.array([k.size for k in kps], dtype=np.float32)
    if len(pts) > cfg.MAX_KEYPOINTS:
        idx = np.argsort(sizes)[::-1][:cfg.MAX_KEYPOINTS]
        pts = pts[idx]
        sizes = sizes[idx]
    return pts, sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Regenerate all cache files')
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_path(cfg.CALIBRATION_FILE, "Calibration file")
    check_path(cfg.LEFT_IMAGE_DIR, "Left image dir")
    check_path(cfg.RIGHT_IMAGE_DIR, "Right image dir")

    calib = np.load(cfg.CALIBRATION_FILE)
    map1_l = calib['map1_left']
    map2_l = calib['map2_left']
    map1_r = calib['map1_right']
    map2_r = calib['map2_right']

    os.makedirs(cfg.FEATURE_CACHE_DIR, exist_ok=True)

    print(f"[Cache] Loading DINOv3 from: {cfg.DINO_LOCAL_PATH}")
    try:
        dino = AutoModel.from_pretrained(cfg.DINO_LOCAL_PATH, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv3 model from {cfg.DINO_LOCAL_PATH}. "
            f"Please ensure the model files exist. Error: {e}"
        )
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False

    patch_size = dino.config.patch_size
    feat_dim = dino.config.hidden_size
    print(f"[Cache] DINOv3: patch_size={patch_size}, feat_dim={feat_dim}")

    left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
    print(f"[Cache] Found {len(left_images)} left images")

    cached = 0
    skipped = 0

    for l_path in tqdm(left_images, desc="Precomputing"):
        basename = os.path.splitext(os.path.basename(l_path))[0]
        cache_path = os.path.join(cfg.FEATURE_CACHE_DIR, f"{basename}.pt")

        if os.path.exists(cache_path) and not args.force:
            skipped += 1
            continue

        filename = os.path.basename(l_path)
        if "left" in filename:
            r_name = filename.replace("left", "right")
        elif "Left" in filename:
            r_name = filename.replace("Left", "Right")
        else:
            r_name = filename
        r_path = os.path.join(cfg.RIGHT_IMAGE_DIR, r_name)

        if not os.path.exists(r_path):
            continue

        l_raw = cv2.imread(l_path, 0)
        r_raw = cv2.imread(r_path, 0)
        if l_raw is None or r_raw is None:
            continue

        l_rect = cv2.remap(l_raw, map1_l, map2_l, cv2.INTER_LINEAR)
        r_rect = cv2.remap(r_raw, map1_r, map2_r, cv2.INTER_LINEAR)

        kp_l, sc_l = detect_blobs(l_rect, cfg)
        kp_r, sc_r = detect_blobs(r_rect, cfg)

        _, mask = cv2.threshold(l_rect, cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        l_gray_t = torch.from_numpy(l_rect).to(torch.uint8)
        r_gray_t = torch.from_numpy(r_rect).to(torch.uint8)
        mask_t = torch.from_numpy(mask).to(torch.uint8)

        l_rgb = cv2.cvtColor(l_rect, cv2.COLOR_GRAY2RGB)
        r_rgb = cv2.cvtColor(r_rect, cv2.COLOR_GRAY2RGB)
        l_rgb_t = torch.from_numpy(l_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        r_rgb_t = torch.from_numpy(r_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        l_rgb_padded, _ = pad_to_patch(l_rgb_t, patch_size)
        r_rgb_padded, _ = pad_to_patch(r_rgb_t, patch_size)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                l_feat = dino(l_rgb_padded.to(device)).last_hidden_state
                r_feat = dino(r_rgb_padded.to(device)).last_hidden_state

        _, _, H_pad, W_pad = l_rgb_padded.shape
        n_patches_h = H_pad // patch_size
        n_patches_w = W_pad // patch_size
        l_feat_map = l_feat[:, -(n_patches_h * n_patches_w):].transpose(1, 2).reshape(
            1, feat_dim, n_patches_h, n_patches_w
        )
        r_feat_map = r_feat[:, -(n_patches_h * n_patches_w):].transpose(1, 2).reshape(
            1, feat_dim, n_patches_h, n_patches_w
        )

        cache_data = {
            'feat_left': l_feat_map.cpu().half().squeeze(0),
            'feat_right': r_feat_map.cpu().half().squeeze(0),
            'keypoints_left': torch.from_numpy(kp_l),
            'scores_left': torch.from_numpy(sc_l),
            'keypoints_right': torch.from_numpy(kp_r),
            'scores_right': torch.from_numpy(sc_r),
            'left_gray': l_gray_t,
            'right_gray': r_gray_t,
            'mask': mask_t,
        }

        torch.save(cache_data, cache_path)
        cached += 1

    print(f"\n[Cache] Done! Cached: {cached}, Skipped (existing): {skipped}")
    print(f"[Cache] Cache dir: {cfg.FEATURE_CACHE_DIR}")


if __name__ == "__main__":
    main()
