"""
DINOv3 Wave Reconstruction - Dataset
======================================
Stereo image dataset with rectification and train/val splitting.
"""

import os
import sys
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import check_path


class RectifiedWaveStereoDataset(Dataset):
    """
    Dataset that loads stereo image pairs, applies rectification maps,
    and returns tensors ready for the model.

    The dataset auto-detects image resolution from calibration maps.
    Train/val split is 90/10 by default.
    """

    def __init__(self, cfg, is_validation=False):
        self.cfg = cfg

        # Resolve calibration file path
        calib_path = cfg.CALIBRATION_FILE
        check_path(calib_path, "标定文件")

        # Discover images
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"[Dataset] No images found in {cfg.LEFT_IMAGE_DIR}")

        # Load calibration
        print(f"[Dataset] Loading Calibration: {calib_path}")
        calib = np.load(calib_path)
        self.map1_l = calib['map1_left']
        self.map2_l = calib['map2_left']
        self.map1_r = calib['map1_right']
        self.map2_r = calib['map2_right']
        self.Q_base = calib['Q'].astype(np.float32)

        # Auto-detect resolution from maps
        h, w = self.map1_l.shape[:2]
        if cfg.IMAGE_WIDTH == 0:
            cfg.IMAGE_WIDTH = w
            cfg.IMAGE_HEIGHT = h
            print(f"[Dataset] Full Resolution Mode: {w}x{h}")

        # Train/Val split
        indices = np.arange(len(self.left_images))
        split = int(len(indices) * 0.9)
        self.indices = indices[split:] if is_validation else indices[:split]
        split_name = "Validation" if is_validation else "Training"
        print(f"[Dataset] {split_name}: {len(self.indices)} images")

    def __len__(self):
        return len(self.indices)

    def get_Q_tensor(self):
        return torch.from_numpy(self.Q_base).float()

    def __getitem__(self, idx):
        idx = self.indices[idx]
        l_path = self.left_images[idx]
        filename = os.path.basename(l_path)

        # Infer right image filename
        if "left" in filename:
            r_name = filename.replace("left", "right")
        elif "Left" in filename:
            r_name = filename.replace("Left", "Right")
        else:
            r_name = filename
        r_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, r_name)

        if not os.path.exists(r_path):
            return None

        # Load and rectify
        l_raw = cv2.imread(l_path, 0)
        r_raw = cv2.imread(r_path, 0)
        if l_raw is None or r_raw is None:
            return None

        l_rect = cv2.remap(l_raw, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        r_rect = cv2.remap(r_raw, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        # Threshold mask on left image
        _, mask = cv2.threshold(l_rect, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Convert to tensors
        l_tensor = torch.from_numpy(l_rect).float().unsqueeze(0) / 255.0
        r_tensor = torch.from_numpy(r_rect).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        # RGB tensors for DINO (gray -> 3-channel)
        l_rgb = cv2.cvtColor(l_rect, cv2.COLOR_GRAY2RGB)
        r_rgb = cv2.cvtColor(r_rect, cv2.COLOR_GRAY2RGB)
        l_rgb_t = torch.from_numpy(l_rgb.transpose(2, 0, 1)).float() / 255.0
        r_rgb_t = torch.from_numpy(r_rgb.transpose(2, 0, 1)).float() / 255.0

        Q = self.get_Q_tensor()
        return {
            'left_gray': l_tensor,
            'right_gray': r_tensor,
            'left_rgb': l_rgb_t,
            'right_rgb': r_rgb_t,
            'mask': mask_tensor,
            'Q': Q,
        }


def stereo_collate_fn(batch):
    """Custom collate that filters out None samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
