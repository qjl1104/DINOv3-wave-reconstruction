"""
DINOv3 Wave Reconstruction - Model Definitions
================================================
Single source of truth for all model classes.
Train and inference scripts import from here.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None
    import sys
    print("\n" + "=" * 50)
    print("错误: 缺少 transformers 库。请运行: pip install transformers")
    print("=" * 50 + "\n")
    sys.exit(1)


class SparseKeypointDetector(nn.Module):
    """Blob-based sparse keypoint detector using OpenCV SimpleBlobDetector."""

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
        self.det = cv2.SimpleBlobDetector_create(p)

    def forward(self, img, mask):
        """
        Detect keypoints in each image of the batch.

        Args:
            img: [B, 1, H, W] grayscale image tensor
            mask: [B, 1, H, W] binary mask (currently unused but kept for API)

        Returns:
            keypoints: [B, N, 2] padded keypoint coordinates (x, y)
            scores: [B, N] keypoint sizes as scores
        """
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
        max_l = max(len(k) for k in kpts)
        if max_l == 0:
            max_l = 1
        k_pad = [torch.cat([k, torch.zeros(max_l - len(k), 2, device=img.device)], 0) for k in kpts]
        s_pad = [torch.cat([s, torch.zeros(max_l - len(s), device=img.device)], 0) for s in scores]
        return torch.stack(k_pad), torch.stack(s_pad)


class DINOv3FeatureExtractor(nn.Module):
    """DINOv2-based feature extractor with grid sampling at keypoint locations."""

    def __init__(self, cfg):
        super().__init__()
        try:
            print(f"[DINO] Loading local model: {cfg.DINO_LOCAL_PATH}")
            self.dino = AutoModel.from_pretrained(cfg.DINO_LOCAL_PATH, local_files_only=True)
        except Exception:
            print("[Warning] 本地加载失败, 尝试从 HuggingFace 下载 dinov2-base...")
            self.dino = AutoModel.from_pretrained("facebook/dinov2-base")
        for p in self.dino.parameters():
            p.requires_grad = False
        self.feat_dim = self.dino.config.hidden_size
        self.patch = self.dino.config.patch_size

    def forward(self, img, kpts):
        """
        Extract DINO features at keypoint locations via grid sampling.

        Args:
            img: [B, 3, H, W] RGB image tensor
            kpts: [B, N, 2] keypoint coordinates (x, y)

        Returns:
            descriptors: [B, N, feat_dim]
        """
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=img.is_cuda):
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
        desc = F.grid_sample(feat.float(), grid, align_corners=True, padding_mode='border')
        return desc.squeeze(3).transpose(1, 2)


class SparseMatchingStereoModel(nn.Module):
    """
    Sparse stereo matching model: DINO features + positional encoding + Transformer matcher.

    Training mode (apply_epipolar_mask=False):
        Epipolar constraint is handled by the loss function, not the model.
    Inference mode (apply_epipolar_mask=True):
        Epipolar mask is applied before softmax to enforce geometric consistency.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.det = SparseKeypointDetector(cfg)
        self.ext = DINOv3FeatureExtractor(cfg)
        self.proj = nn.Linear(2, cfg.FEATURE_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.FEATURE_DIM, nhead=cfg.NUM_HEADS, batch_first=True
        )
        self.trans = nn.TransformerEncoder(layer, num_layers=cfg.NUM_ATTENTION_LAYERS)
        self.out_proj = nn.Linear(cfg.FEATURE_DIM, cfg.FEATURE_DIM)

    def forward(self, lg, rg, lrgb, rrgb, mask, apply_epipolar_mask=False):
        """
        Forward pass for stereo matching.

        Args:
            lg: left grayscale [B, 1, H, W]
            rg: right grayscale [B, 1, H, W]
            lrgb: left RGB [B, 3, H, W]
            rrgb: right RGB [B, 3, H, W]
            mask: binary mask [B, 1, H, W]
            apply_epipolar_mask: apply epipolar constraint before softmax (for inference)

        Returns:
            dict with keys:
                keypoints_left: [B, N, 2]
                scores_left: [B, N]
                keypoints_right_pred: [B, N, 2] predicted right keypoints
                disparity: [B, N] = left_x - predicted_right_x
                match_scores: [B, N, M] raw similarity scores
        """
        kpl, sl = self.det(lg, mask)
        kpr, sr = self.det(rg, torch.ones_like(rg))

        # --- Sparse Ablation: randomly drop left keypoints ---
        if self.cfg.KEEP_RATIO < 1.0 and not self.training:
            B_kp, N_kp, _ = kpl.shape
            kpl_new = torch.zeros_like(kpl)
            sl_new = torch.zeros_like(sl)
            for b in range(B_kp):
                valid_mask = sl[b] > 0
                valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                num_valid = len(valid_idx)
                if num_valid > 0:
                    keep_count = max(1, int(num_valid * self.cfg.KEEP_RATIO))
                    perm = torch.randperm(num_valid, device=kpl.device)[:keep_count]
                    kept_idx = valid_idx[perm]
                    kpl_new[b, :keep_count] = kpl[b, kept_idx]
                    sl_new[b, :keep_count] = sl[b, kept_idx]
            kpl = kpl_new
            sl = sl_new

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

        if apply_epipolar_mask:
            mask_geo = (kpl[:, :, 1].unsqueeze(2) - kpr[:, :, 1].unsqueeze(1)).abs() < self.cfg.EPIPOLAR_THRESHOLD
            scores = scores.masked_fill(~mask_geo, -1e9)

        probs = F.softmax(scores, dim=-1)
        x_right_ex = (probs * kpr[:, :, 0].unsqueeze(1)).sum(dim=2)
        y_right_ex = (probs * kpr[:, :, 1].unsqueeze(1)).sum(dim=2)
        kp_right_pred = torch.stack([x_right_ex, y_right_ex], dim=-1)
        disparity = kpl[:, :, 0] - x_right_ex

        return {
            'keypoints_left': kpl,
            'scores_left': sl,
            'keypoints_right_pred': kp_right_pred,
            'disparity': disparity,
            'match_scores': scores,
        }
