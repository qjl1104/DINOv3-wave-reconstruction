"""
DINOv3 Wave Reconstruction - Loss Functions
=============================================
PINN (Physics-Informed Neural Network) loss with:
  - Photometric loss (patch-based + intensity penalty)
  - Disparity regularization (negative disparity penalty)
  - Physics constraints (smoothness, slope, zero-mean)

Note: Epipolar constraint is enforced by architecture (same-row matching),
so no explicit epipolar y-diff loss is needed.
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
        self.depth_min = cfg.DEPTH_MIN
        self.depth_max = cfg.DEPTH_MAX
        self.slope_threshold = cfg.SLOPE_THRESHOLD
        self.knn_k = cfg.KNN_K
        self.max_pinn_points = cfg.MAX_PINN_POINTS

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
            zero = points_3d.sum() * 0.0
            return (zero, zero, zero)

        loss_smooth = 0.0
        loss_slope = 0.0
        loss_zeromean = 0.0
        valid_batches = 0

        for b in range(B):
            p3d = points_3d[b]
            sc = scores[b]
            valid_mask = (p3d[:, 2] > self.depth_min) & (p3d[:, 2] < self.depth_max) & (sc > 0.1)
            p_valid = p3d[valid_mask]

            if p_valid.shape[0] < 10:
                continue

            p_valid_m = p_valid / 1000.0
            xy_m = p_valid_m[:, [0, 2]]
            height_m = p_valid_m[:, 1]

            if p_valid_m.shape[0] > self.max_pinn_points:
                perm = torch.randperm(p_valid_m.shape[0], device=p_valid_m.device)[:self.max_pinn_points]
                xy_m = xy_m[perm]
                height_m = height_m[perm]

            dist_matrix = torch.cdist(xy_m, xy_m)
            K = self.knn_k
            if xy_m.shape[0] <= K:
                K = xy_m.shape[0] - 1

            dists, indices = dist_matrix.topk(K + 1, largest=False, dim=1)
            neighbor_indices = indices[:, 1:]
            neighbor_dists = torch.clamp(dists[:, 1:], min=0.001)
            neighbor_heights = height_m[neighbor_indices]

            local_mean = neighbor_heights.mean(dim=1)
            l_smooth = F.smooth_l1_loss(height_m, local_mean, beta=0.01)

            delta_h = (neighbor_heights - height_m.unsqueeze(1)).abs()
            slopes = delta_h / neighbor_dists
            slope_penalty = F.relu(slopes - self.slope_threshold)
            l_slope = slope_penalty.mean()

            l_zeromean = height_m.mean().abs()

            loss_smooth += l_smooth
            loss_slope += l_slope
            loss_zeromean += l_zeromean
            valid_batches += 1

        if valid_batches == 0:
            zero = points_3d.sum() * 0.0
            return (zero, zero, zero)
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
            return patches_l.sum() * 0.0

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
            return patches_l.sum() * 0.0
        loss = (F.smooth_l1_loss(val_l, val_r, reduction='none') * weights).sum() / weight_sum
        return loss

    def forward(self, lg, rg, kpl, kpr, scores, Q, correlation_probs=None,
                disparity=None, kpr_actual=None, disparity_rev=None):
        """计算所有损失。

        Args:
            correlation_probs: Sinkhorn 软分配矩阵列表，每个元素 [n_kp, Wf]。
            disparity: 模型预测的左→右视差 [B, N]，用于 NCC 验证和视差范围先验。
            kpr_actual: 右图实际关键点 [B, N_r, 2]，用于左右一致性。
            disparity_rev: 右→左反向视差 [B, N_r]，用于左右一致性。
        """
        l_disp = self.compute_disp_loss(kpl, kpr, scores)
        l_smooth, l_slope, l_zeromean = self.compute_pinn(kpl, kpr, scores, Q)

        l_corr = torch.tensor(0.0, device=kpl.device)
        if correlation_probs is not None and len(correlation_probs) > 0:
            l_corr = self.compute_correlation_loss(correlation_probs)

        # 反捷径损失
        l_ncc = torch.tensor(0.0, device=kpl.device)
        l_range = torch.tensor(0.0, device=kpl.device)
        l_lr = torch.tensor(0.0, device=kpl.device)

        if disparity is not None:
            l_ncc = self.compute_ncc_match_loss(lg, rg, kpl, disparity, scores)
            l_range = self.compute_disp_range_loss(disparity, scores)

        if disparity is not None and disparity_rev is not None and kpr_actual is not None:
            l_lr = self.compute_lr_consistency_loss(
                kpl, disparity, kpr_actual, disparity_rev, scores
            )

        return l_disp, l_smooth, l_slope, l_zeromean, l_corr, l_ncc, l_range, l_lr

    def compute_photometric(self, lg, rg, kpl, kpr, scores):
        disp = kpl[..., 0] - kpr[..., 0]
        weight_sum = scores.sum()
        neg_disp_penalty = F.relu(-disp) * 0.1
        if weight_sum > 1e-4:
            l_disp = (neg_disp_penalty * scores).sum() / weight_sum
        else:
            l_disp = neg_disp_penalty.mean()

        patches_l = self.sample_patches(lg, kpl, self.patch_size)
        patches_r = self.sample_patches(rg, kpr, self.patch_size)
        l_masked = self.soft_photometric_loss(patches_l, patches_r, scores)
        l_intensity = self.intensity_penalty(patches_l, patches_r, scores)
        l_photo = l_masked + l_intensity
        return l_photo, l_disp

    def compute_disp_loss(self, kpl, kpr, scores):
        """视差正则化损失：惩罚负视差（交叉匹配）。"""
        disp = kpl[..., 0] - kpr[..., 0]
        weight_sum = scores.sum()
        neg_disp_penalty = F.relu(-disp) * 0.1
        if weight_sum > 1e-4:
            l_disp = (neg_disp_penalty * scores).sum() / weight_sum
        else:
            l_disp = neg_disp_penalty.mean()
        return l_disp

    def compute_correlation_loss(self, prob_list):
        """相关体损失：熵 + 峰度，鼓励 Sinkhorn 软分配具有清晰峰值。

        Args:
            prob_list: 软分配矩阵列表，每个元素 [n_kp, Wf]，行归一化（每行和为1）。

        Returns:
            平均损失（熵 + 0.5 * 峰度）。
        """
        if len(prob_list) == 0:
            device = prob_list[0].device if prob_list else torch.device('cpu')
            return torch.tensor(0.0, device=device)

        l_entropy = 0.0
        l_peakiness = 0.0
        n = 0

        for prob in prob_list:
            prob_f = prob.float()
            n_kp = prob_f.shape[0]
            if n_kp == 0:
                continue

            # 熵损失：低熵 → 尖锐峰值，避免均匀分布
            log_prob = torch.log(prob_f + 1e-8)
            entropy = -(prob_f * log_prob).sum(dim=-1).mean()
            l_entropy += entropy

            # 峰度损失：第二高峰 / 第一高峰，越小越尖锐
            if prob_f.shape[1] >= 2:
                top2 = torch.topk(prob_f, k=2, dim=-1).values
                peakiness = (top2[:, 1] / (top2[:, 0] + 1e-8)).mean()
                l_peakiness += peakiness

            n += 1

        if n == 0:
            return torch.tensor(0.0, device=prob_list[0].device)

        return (l_entropy + 0.5 * l_peakiness) / n

    def compute_pinn(self, kpl, kpr, scores, Q):
        mask_final = (scores > 0.1)
        if mask_final.sum() < 10:
            zero = kpl.sum() * 0.0
            return (zero, zero, zero)

        kpl_f = kpl.float()
        kpr_f = kpr.float()
        disp_f = (kpl_f[..., 0] - kpr_f[..., 0])
        Q_f = Q.float()
        points_3d = self.disparity_to_3d(kpl_f, disp_f, Q_f)
        l_smooth, l_slope, l_zeromean = self.compute_pinn_loss(points_3d, scores.float())
        return l_smooth, l_slope, l_zeromean

    def compute_ncc_match_loss(self, lg, rg, kpl, disparity, scores):
        """NCC 匹配验证损失：在模型预测的匹配位置计算归一化互相关。

        核心思路：如果模型预测的视差正确，左图 patch 和右图对应 patch 应该高度相似（NCC≈1）。
        如果模型走捷径预测 disp≈0，左右 patch 对应不同的小圆片，NCC≈0 → 被惩罚。

        重要修复：零视差时 NCC 退化（同位置采样 → NCC=1 反而奖励捷径）。
        所以只在 disp >= DISP_MIN_PRIOR 时计算 NCC，零视差交由 disp_range_loss 惩罚。
        """
        B, N, _ = kpl.shape
        device = kpl.device
        ps = self.patch_size

        # 在 fp32 下计算以保证数值稳定性
        with torch.amp.autocast('cuda', enabled=False):
            lg_f = lg.float()
            rg_f = rg.float()
            kpl_f = kpl.float()
            disp_f = disparity.float()
            scores_f = scores.float()

            # 构造右图匹配位置：right_x = left_x - disp
            kpr_pred = kpl_f.clone()
            kpr_pred[..., 0] = kpl_f[..., 0] - disp_f

            # 采样左右 patch
            patches_l = self.sample_patches(lg_f, kpl_f, ps)  # [B, C, N, ps, ps]
            patches_r = self.sample_patches(rg_f, kpr_pred, ps)

            # 展开并归一化（NCC = 归一化后的点积）
            C = patches_l.shape[1]
            pl = patches_l.reshape(B, C, N, -1)  # [B, C, N, ps*ps]
            pr = patches_r.reshape(B, C, N, -1)

            # 逐 patch 去均值 + L2 归一化
            pl_norm = F.normalize(pl - pl.mean(dim=-1, keepdim=True), dim=-1)
            pr_norm = F.normalize(pr - pr.mean(dim=-1, keepdim=True), dim=-1)

            # NCC 值域 [-1, 1]，越接近 1 越匹配
            ncc = (pl_norm * pr_norm).sum(dim=-1).mean(dim=1)  # [B, N]

            # 损失 = 1 - NCC（加权 by scores）
            loss_per_pt = (1.0 - ncc)  # [B, N]

            # 关键：只在 disp >= DISP_MIN_PRIOR 时启用 NCC 损失
            # disp < DISP_MIN_PRIOR 时返回 0（交由 disp_range_loss 惩罚）
            # 否则零视差时 NCC=1 反而 reward 捷径
            valid_ncc_mask = (disp_f >= self.cfg.DISP_MIN_PRIOR).float()
            effective_weight = scores_f * valid_ncc_mask

            weight_sum = effective_weight.sum()
            if weight_sum < 1e-4:
                return lg_f.sum() * 0.0

            loss = (loss_per_pt * effective_weight).sum() / weight_sum

        return loss

    def compute_disp_range_loss(self, disparity, scores):
        """视差范围先验损失：惩罚视差超出物理合理范围。

        基于 Q 矩阵: fB ≈ 3,718,679
        Z=2000mm → d≈1859, Z=15000mm → d≈248
        合理范围 [DISP_MIN_PRIOR, DISP_MAX_PRIOR]
        """
        disp_min = self.cfg.DISP_MIN_PRIOR
        disp_max = self.cfg.DISP_MAX_PRIOR

        # 软惩罚：超出范围的部分用 relu
        penalty_low = F.relu(disp_min - disparity)    # 视差太小 → 太远
        penalty_high = F.relu(disparity - disp_max)   # 视差太大 → 太近

        penalty = penalty_low + penalty_high

        weight_sum = scores.sum()
        if weight_sum < 1e-4:
            return disparity.sum() * 0.0

        return (penalty * scores).sum() / weight_sum

    def compute_lr_consistency_loss(self, kpl, disp_lr, kpr, disp_rl, scores):
        """左右一致性损失：正向和反向匹配应该一致。

        对于左图关键点 i（视差 disp_lr[i]）：
          → 右图匹配位置: x_r = kpl[i].x - disp_lr[i]
          → 找最近的右图关键点 j
          → 反向视差 disp_rl[j] 应满足: disp_lr[i] + disp_rl[j] ≈ 0

        注意：反向视差的符号约定是 right_col - left_match_col，
        所以一致时 disp_lr + disp_rl ≈ 0。
        """
        B, N, _ = kpl.shape
        device = kpl.device
        total_loss = 0.0
        valid_batches = 0

        for b in range(B):
            kp_l = kpl[b]  # [N, 2]
            d_lr = disp_lr[b]  # [N]
            sc = scores[b]  # [N]
            kp_r = kpr[b]  # [N_r, 2]
            d_rl = disp_rl[b]  # [N_r]

            valid = sc > 0.1
            if valid.sum() < 5 or kp_r.shape[0] < 5:
                continue

            kp_l_v = kp_l[valid]
            d_lr_v = d_lr[valid]

            # 左图关键点在右图的预测位置
            right_x_pred = kp_l_v[:, 0] - d_lr_v  # [M]

            # 找每个预测位置最近的右图关键点
            # right_x_pred: [M], kp_r[:, 0]: [N_r]
            diff = right_x_pred.unsqueeze(1) - kp_r[:, 0].unsqueeze(0)  # [M, N_r]
            nearest_idx = diff.abs().argmin(dim=1)  # [M]

            # 取对应右图关键点的反向视差
            d_rl_nearest = d_rl[nearest_idx]  # [M]

            # 一致性: disp_lr + disp_rl ≈ 0
            consistency = (d_lr_v + d_rl_nearest).abs()

            total_loss += consistency.mean()
            valid_batches += 1

        if valid_batches == 0:
            return kpl.sum() * 0.0
        return total_loss / valid_batches
