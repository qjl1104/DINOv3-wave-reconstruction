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
from tqdm import tqdm

from config import check_path


class RectifiedWaveStereoDataset(Dataset):
    """
    Dataset that loads stereo image pairs, applies rectification maps,
    and returns tensors ready for the model.

    The dataset auto-detects image resolution from calibration maps.
    Train/val split is 90/10 by default.

    When USE_FEATURE_CACHE is True and cache files exist, returns
    precomputed DINO features + blob keypoints (fast path).
    """

    def __init__(self, cfg, is_validation=False):
        self.cfg = cfg
        self.use_cache = cfg.USE_FEATURE_CACHE

        calib_path = cfg.CALIBRATION_FILE
        check_path(calib_path, "标定文件")

        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            sys.exit(f"[Dataset] No images found in {cfg.LEFT_IMAGE_DIR}")

        print(f"[Dataset] Loading Calibration: {calib_path}")
        calib = np.load(calib_path)
        self.map1_l = calib['map1_left']
        self.map2_l = calib['map2_left']
        self.map1_r = calib['map1_right']
        self.map2_r = calib['map2_right']
        self.Q_base = calib['Q'].astype(np.float32)

        h, w = self.map1_l.shape[:2]
        if cfg.IMAGE_WIDTH == 0:
            cfg.IMAGE_WIDTH = w
            cfg.IMAGE_HEIGHT = h
            print(f"[Dataset] Full Resolution Mode: {w}x{h}")

        indices = np.arange(len(self.left_images))
        split = int(len(indices) * 0.9)
        self.indices = indices[split:] if is_validation else indices[:split]
        split_name = "Validation" if is_validation else "Training"

        if self.use_cache:
            cache_dir = cfg.FEATURE_CACHE_DIR
            valid_indices = []
            for idx in self.indices:
                basename = os.path.splitext(os.path.basename(self.left_images[idx]))[0]
                cache_path = os.path.join(cache_dir, f"{basename}.pt")
                if os.path.exists(cache_path):
                    valid_indices.append(idx)
            if len(valid_indices) < len(self.indices):
                missing = len(self.indices) - len(valid_indices)
                print(f"[Dataset] WARNING: {missing} images have no cache, using {len(valid_indices)} cached images")
                if len(valid_indices) == 0:
                    print("[Dataset] No cache found! Run: python precompute_cache.py")
                    print("[Dataset] Falling back to non-cached mode.")
                    self.use_cache = False
                else:
                    self.indices = np.array(valid_indices)

            if self.use_cache:
                self.cache_paths = {}
                missing_cache = 0
                for idx in self.indices:
                    basename = os.path.splitext(os.path.basename(self.left_images[idx]))[0]
                    cache_path = os.path.join(cache_dir, f"{basename}.pt")
                    if os.path.exists(cache_path):
                        self.cache_paths[idx] = cache_path
                    else:
                        missing_cache += 1
                if missing_cache > 0:
                    print(f"[Dataset] WARNING: {missing_cache} cache files missing, will skip those samples")
                self._cache_dir = cache_dir

                if is_validation:
                    print(f"[Dataset] {split_name}: {len(self.indices)} images (lazy-load from disk)")
                    self.mem_cache = None
                else:
                    print(f"[Dataset] {split_name}: {len(self.indices)} images — loading features to RAM...")
                    self.mem_cache = {}
                    for idx in tqdm(self.indices, desc=f"Loading {split_name}", unit="img"):
                        cache_path = self.cache_paths.get(idx)
                        if cache_path is None:
                            continue
                        full_cache = torch.load(cache_path, map_location='cpu', weights_only=False)
                        self.mem_cache[idx] = {
                            'feat_left': full_cache['feat_left'].clone(),
                            'feat_right': full_cache['feat_right'].clone(),
                            'keypoints_left': full_cache['keypoints_left'].clone(),
                            'scores_left': full_cache['scores_left'].clone(),
                            'keypoints_right': full_cache['keypoints_right'].clone(),
                            'scores_right': full_cache['scores_right'].clone(),
                            'left_gray': full_cache['left_gray'].clone(),
                            'right_gray': full_cache['right_gray'].clone(),
                            'mask': full_cache['mask'].clone(),
                        }
                        del full_cache
                    print(f"[Dataset] {split_name}: {len(self.mem_cache)} samples loaded to RAM")
        else:
            print(f"[Dataset] {split_name}: {len(self.indices)} images")

    def __len__(self):
        return len(self.indices)

    def get_Q_tensor(self):
        return torch.from_numpy(self.Q_base).float()

    def __getitem__(self, idx):
        idx = self.indices[idx]
        l_path = self.left_images[idx]
        filename = os.path.basename(l_path)
        basename = os.path.splitext(filename)[0]

        if self.use_cache:
            return self._load_cached(idx, l_path, basename)
        return self._load_raw(idx, l_path, filename)

    def _load_cached(self, idx, l_path, basename):
        if self.mem_cache is not None:
            cached_feats = self.mem_cache.get(idx)
            if cached_feats is None:
                return None
            feat_left = cached_feats['feat_left']
            feat_right = cached_feats['feat_right']
            kpl = cached_feats['keypoints_left']
            sl = cached_feats['scores_left']
            kpr = cached_feats['keypoints_right']
            sr = cached_feats['scores_right']
            l_gray = cached_feats['left_gray']
            r_gray = cached_feats['right_gray']
            mask_np = cached_feats['mask']
        else:
            cache_path = self.cache_paths.get(idx)
            if cache_path is None:
                return None
            cache = torch.load(cache_path, map_location='cpu', weights_only=False)
            feat_left = cache['feat_left'].clone()
            feat_right = cache['feat_right'].clone()
            kpl = cache['keypoints_left'].clone()
            sl = cache['scores_left'].clone()
            kpr = cache['keypoints_right'].clone()
            sr = cache['scores_right'].clone()
            l_gray = cache['left_gray']
            r_gray = cache['right_gray']
            mask_np = cache['mask']

        # 直接使用缓存中的灰度图和 mask，避免重复 IO 和 remap
        l_tensor = l_gray.float().unsqueeze(0) / 255.0
        r_tensor = r_gray.float().unsqueeze(0) / 255.0
        mask_tensor = mask_np.float().unsqueeze(0) / 255.0

        Q = self.get_Q_tensor()
        return {
            'left_gray': l_tensor,
            'right_gray': r_tensor,
            'mask': mask_tensor,
            'Q': Q,
            'cached': True,
            'feat_left': feat_left,
            'feat_right': feat_right,
            'keypoints_left': kpl,
            'scores_left': sl,
            'keypoints_right': kpr,
            'scores_right': sr,
        }

    def _load_raw(self, idx, l_path, filename):

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
            'cached': False,
        }


def stereo_collate_fn(batch):
    """Custom collate that filters out None samples, handles mixed cached/raw."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    is_cached = batch[0].get('cached', False)
    if is_cached:
        result = {}
        for key in batch[0]:
            if key == 'cached':
                result[key] = True
            elif isinstance(batch[0][key], torch.Tensor):
                if key in ('keypoints_left', 'scores_left', 'keypoints_right', 'scores_right'):
                    max_len = max(b[key].shape[0] for b in batch)
                    padded = []
                    for b in batch:
                        t = b[key]
                        if t.shape[0] < max_len:
                            pad_shape = list(t.shape)
                            pad_shape[0] = max_len - t.shape[0]
                            t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
                        padded.append(t)
                    result[key] = torch.stack(padded)
                else:
                    result[key] = torch.stack([b[key] for b in batch])
        return result
    return torch.utils.data.dataloader.default_collate(batch)
