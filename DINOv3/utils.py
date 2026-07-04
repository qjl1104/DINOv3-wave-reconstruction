"""
DINOv3 Wave Reconstruction - Shared Utilities
===============================================
Common functions used across train, inference, ablation, and diagnostic scripts.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F


def pad_to_patch_size(*tensors, patch_size=14):
    """Pad tensors so spatial dims are divisible by patch_size."""
    h, w = tensors[0].shape[-2:]
    padh = (patch_size - h % patch_size) % patch_size
    padw = (patch_size - w % patch_size) % patch_size
    if padh > 0 or padw > 0:
        return [F.pad(t, (0, padw, 0, padh)) for t in tensors]
    return list(tensors)


def infer_right_path(left_path, right_dir):
    """Infer the right image path from a left image path."""
    basename = os.path.basename(left_path)
    if "left" in basename:
        r_name = basename.replace("left", "right")
    elif "Left" in basename:
        r_name = basename.replace("Left", "Right")
    else:
        r_name = basename
    return os.path.join(right_dir, r_name)


def reproject_to_3d(keypoints, disparity, Q):
    """
    Reproject 2D keypoints + disparity to 3D using Q matrix.

    Args:
        keypoints: [N, 2] numpy array (x, y)
        disparity: [N] numpy array
        Q: [4, 4] numpy array

    Returns:
        points_3d: [M, 3] numpy array (only valid points where w > 1e-6)
    """
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
    pts = np.zeros((N, 3))
    pts[valid] = projected[valid, :3] / w[valid, None]
    return pts[valid]


def downsample_keypoints(kp, scores, keep_ratio):
    """
    Randomly drop keypoints to simulate sparsity.

    Args:
        kp: [B, N, 2] keypoint tensor
        scores: [B, N] score tensor
        keep_ratio: float in (0, 1], fraction of valid keypoints to keep

    Returns:
        (kp_new, sc_new) with same shape, zero-padded
    """
    if keep_ratio >= 1.0:
        return kp, scores
    B, N, _ = kp.shape
    kp_new = torch.zeros_like(kp)
    sc_new = torch.zeros_like(scores)
    for b in range(B):
        valid = scores[b] > 0
        valid_idx = valid.nonzero(as_tuple=True)[0]
        n_valid = len(valid_idx)
        if n_valid > 0:
            keep = max(1, int(n_valid * keep_ratio))
            perm = torch.randperm(n_valid, device=kp.device)[:keep]
            kept = valid_idx[perm]
            kp_new[b, :keep] = kp[b, kept]
            sc_new[b, :keep] = scores[b, kept]
    return kp_new, sc_new


def load_model_checkpoint(model, path, device, strict=False):
    """
    Load a checkpoint into a model, supporting both old (pure state_dict)
    and new (dict with 'model_state_dict') formats.

    Returns:
        The loaded checkpoint dict (or state_dict).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    return checkpoint
