"""
DINOv3 Wave Reconstruction - Loss Functions
=============================================
PINN (Physics-Informed Neural Network) loss with:
  - Photometric loss (patch-based + intensity penalty)
  - Epipolar loss (y-coordinate alignment)
  - Physics constraints (smoothness, slope, zero-mean)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNPhysicsLoss(nn.Module):
    """Combined photometric + geometric + physics loss for stereo wave reconstruction."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.PATCH_SIZE_PHOTOMETRIC
        self.bright_threshold = 0.02

    def disparity_to_3d(self, keypoints, disparity, Q):
        """Convert 2D keypoints + disparity to 3D points using Q matrix."""
        B, N, _ = keypoints.shape
        device = keypoints.device
        disp_unsqueezed = disparity.unsqueeze(-1)
        ones = torch.ones((B, N, 1), device=device)
        points_4d = torch.cat([keypoints, disp_unsqueezed, ones], dim=-1)
        projected = torch.matmul(points_4d, Q.transpose(1, 2))
        X, Y, Z, W = projected.unbind(-1)
        W = torch.clamp(W, min=1e-6)
        points_3d = torch.stack([X / W, Y / W, Z / W], dim=-1)
        return points_3d

    def compute_pinn_loss(self, points_3d, scores):
        """Compute physics-based losses: smoothness, slope penalty, zero-mean."""
        B, N, _ = points_3d.shape
        device = points_3d.device
        if N < 10:
            return (torch.tensor(0.0, device=device),) * 3

        loss_smooth = 0.0
        loss_slope = 0.0
        loss_zeromean = 0.0
        valid_batches = 0

        for b in range(B):
            p3d = points_3d[b]
            sc = scores[b]
            valid_mask = (p3d[:, 2] > 100) & (p3d[:, 2] < 30000) & (sc > 0.1)
            p_valid = p3d[valid_mask]

            if p_valid.shape[0] < 10:
                continue

            p_valid_m = p_valid / 1000.0
            xy_m = p_valid_m[:, [0, 2]]
            height_m = p_valid_m[:, 1]

            if p_valid_m.shape[0] > 2000:
                perm = torch.randperm(p_valid_m.shape[0])[:2000]
                xy_m = xy_m[perm]
                height_m = height_m[perm]

            dist_matrix = torch.cdist(xy_m, xy_m)
            K = 5
            if xy_m.shape[0] <= K:
                K = xy_m.shape[0] - 1

            dists, indices = dist_matrix.topk(K + 1, largest=False, dim=1)
            neighbor_indices = indices[:, 1:]
            neighbor_dists = torch.clamp(dists[:, 1:], min=0.001)
            neighbor_heights = height_m[neighbor_indices]

            # 平滑损失 (KNN local mean)
            local_mean = neighbor_heights.mean(dim=1)
            l_smooth = F.smooth_l1_loss(height_m, local_mean, beta=0.01)

            # 斜率损失 (penalize slopes > 0.4)
            delta_h = (neighbor_heights - height_m.unsqueeze(1)).abs()
            slopes = delta_h / neighbor_dists
            slope_penalty = F.relu(slopes - 0.4)
            l_slope = slope_penalty.mean()

            # 零均值
            l_zeromean = height_m.mean().abs()

            loss_smooth += l_smooth
            loss_slope += l_slope
            loss_zeromean += l_zeromean
            valid_batches += 1

        if valid_batches == 0:
            return (torch.tensor(0.0, device=device),) * 3
        return (loss_smooth / valid_batches, loss_slope / valid_batches, loss_zeromean / valid_batches)

    def sample_patches(self, image, keypoints, patch_size):
        """Sample image patches at keypoint locations via grid_sample."""
        B, N, _ = keypoints.shape
        device = keypoints.device
        half = patch_size // 2
        xv, yv = torch.meshgrid(
            torch.linspace(-half, half, patch_size),
            torch.linspace(-half, half, patch_size), indexing='ij'
        )
        grid_rel = torch.stack([yv, xv], dim=-1).to(device).view(1, 1, -1, 2)
        kp_expand = keypoints.unsqueeze(2) + grid_rel
        b, c, h, w = image.shape
        kp_norm = kp_expand.clone()
        w = max(w, 2)
        h = max(h, 2)
        kp_norm[..., 0] = 2 * kp_norm[..., 0] / (w - 1) - 1
        kp_norm[..., 1] = 2 * kp_norm[..., 1] / (h - 1) - 1
        kp_norm = kp_norm.view(B, -1, 1, 2)
        patches = F.grid_sample(image, kp_norm, align_corners=True, mode='bilinear', padding_mode='border')
        patches = patches.view(B, c, N, patch_size, patch_size)
        return patches

    def soft_photometric_loss(self, patches_l, patches_r, scores):
        """Weighted photometric loss (bright pixels only)."""
        center_idx = self.patch_size // 2
        center_val = patches_l[:, :, :, center_idx, center_idx]
        is_bright = (center_val > self.bright_threshold).float().squeeze(1)

        diff = (patches_l - patches_r).abs().mean(dim=[1, 3, 4])
        weights = scores * is_bright
        weight_sum = weights.sum()
        if weight_sum < 1e-4:
            return torch.tensor(0.0, device=patches_l.device)

        loss = (diff * weights).sum() / weight_sum
        return loss

    def intensity_penalty(self, patches_l, patches_r, scores):
        """Center-pixel intensity matching penalty."""
        center = self.patch_size // 2
        val_l = patches_l[:, :, :, center, center].squeeze(1)
        val_r = patches_r[:, :, :, center, center].squeeze(1)
        is_bright = (val_l > self.bright_threshold).float()
        weights = scores * is_bright
        weight_sum = weights.sum()
        if weight_sum < 1e-4:
            return torch.tensor(0.0, device=patches_l.device)
        loss = (F.smooth_l1_loss(val_l, val_r, reduction='none') * weights).sum() / weight_sum
        return loss

    def forward(self, lg, rg, kpl, kpr, scores, Q):
        """
        Compute all losses.

        Args:
            lg, rg: left/right grayscale images [B, 1, H, W]
            kpl, kpr: left/right keypoints [B, N, 2]
            scores: keypoint scores [B, N]
            Q: reprojection matrix [B, 4, 4]

        Returns:
            (l_photo, l_epipolar, l_smooth, l_slope, l_zeromean)
        """
        # 1. Epipolar loss
        y_diff = (kpl[..., 1] - kpr[..., 1]).abs()
        weight_sum = scores.sum()
        if weight_sum > 1e-4:
            l_epipolar = (y_diff * scores).sum() / weight_sum
        else:
            l_epipolar = y_diff.mean()

        # 2. Photometric loss
        patches_l = self.sample_patches(lg, kpl, self.patch_size)
        patches_r = self.sample_patches(rg, kpr, self.patch_size)
        l_masked = self.soft_photometric_loss(patches_l, patches_r, scores)
        l_intensity = self.intensity_penalty(patches_l, patches_r, scores)
        l_photo = l_masked + l_intensity

        # 3. PINN physics losses
        mask_final = (scores > 0.1)

        if mask_final.sum() < 10:
            l_smooth, l_slope, l_zeromean = (torch.tensor(0.0, device=lg.device),) * 3
        else:
            # Force float32 for physics computation (fp16 causes overflow on RTX 5080)
            with torch.amp.autocast('cuda', enabled=False):
                kpl_f = kpl.float()
                kpr_f = kpr.float()
                disp_f = (kpl_f[..., 0] - kpr_f[..., 0])
                Q_f = Q.float()
                points_3d = self.disparity_to_3d(kpl_f, disp_f, Q_f)
                l_smooth, l_slope, l_zeromean = self.compute_pinn_loss(points_3d, scores.float())

        return l_photo, l_epipolar, l_smooth, l_slope, l_zeromean
