"""
DINOv3 Wave Reconstruction - Model Definitions
================================================
DINOv3 backbone + 1D Correlation Volume + Soft-argmax disparity regression.
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
    def __init__(self, cfg):
        super().__init__()
        try:
            print(f"[DINO] Loading local model: {cfg.DINO_LOCAL_PATH}", flush=True)
            self.dino = AutoModel.from_pretrained(cfg.DINO_LOCAL_PATH, local_files_only=True)
            print(f"[DINO] Model loaded to CPU", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 model from {cfg.DINO_LOCAL_PATH}. "
                f"Please ensure the model files exist. Error: {e}"
            )
        for p in self.dino.parameters():
            p.requires_grad = False
        self.feat_dim = self.dino.config.hidden_size
        self.patch = self.dino.config.patch_size
        print(f"[DINO] Feature extractor ready: dim={self.feat_dim}, patch={self.patch}", flush=True)

    def forward_dense(self, img):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = self.dino(img).last_hidden_state
        B, _, H, W = img.shape
        n_h = H // self.patch
        n_w = W // self.patch
        feat = out[:, -(n_h * n_w):]
        feat = feat.transpose(1, 2).reshape(B, self.feat_dim, n_h, n_w)
        return feat


class CorrRefinementNet(nn.Module):
    def __init__(self, max_disp_feat=160):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=5, padding=2)
        self.act = nn.GELU()

    def forward(self, corr):
        x = corr.unsqueeze(1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x).squeeze(1)
        return corr + x


class CorrMatchingStereoModel(nn.Module):
    """
    DINOv3-based stereo matching via 1D Correlation Volume + Sinkhorn 匹配。

    Architecture:
      1. DINOv3 backbone extracts dense feature maps [B, C, Hf, Wf]
      2. Feature projection: C -> D (learnable, reduces correlation cost)
      3. For each epipolar row, compute 1D correlation between ALL left keypoints
         and the dense right feature row simultaneously
      4. Sinkhorn optimal transport enforces global consistency:
         - 同一行的关键点不会全部匹配到同一个右图位置
         - 产生近似一对一的行级软分配
      5. Soft-argmax extracts sub-pixel disparity from Sinkhorn assignment

    相比逐点独立 soft-argmax，Sinkhorn 的核心优势：
      - 打破退化：多个相同关键点不能同时匹配到同一位置
      - 全局一致性：利用关键点间的相对位置约束
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.det = SparseKeypointDetector(cfg)
        self.ext = DINOv3FeatureExtractor(cfg)

        feat_dim = self.ext.feat_dim
        proj_dim = cfg.CORR_PROJ_DIM

        self.proj_l = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim, 1),
        )
        self.proj_r = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim, 1),
        )

        self.corr_refine = CorrRefinementNet()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.sinkhorn_eps = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def sinkhorn(cost, epsilon, num_iters=10):
        """
        Sinkhorn 最优传输算法（对数域，fp32 内部计算）。

        Args:
            cost: (n, m) 代价矩阵，值越小匹配越好
            epsilon: 熵正则化系数，越大分配越均匀
            num_iters: Sinkhorn 迭代次数

        Returns:
            P: (n, m) 行归一化的软分配矩阵，每行和为 1
        """
        with torch.amp.autocast('cuda', enabled=False):
            cost32 = cost.float()
            eps = epsilon.float().abs().clamp(min=0.01)

            n, m = cost32.shape
            log_K = -cost32 / eps

            log_a = torch.full((n,), -np.log(n), device=cost32.device, dtype=torch.float32)
            log_b = torch.full((m,), -np.log(m), device=cost32.device, dtype=torch.float32)

            log_u = torch.zeros(n, device=cost32.device, dtype=torch.float32)
            log_v = torch.zeros(m, device=cost32.device, dtype=torch.float32)

            for _ in range(num_iters):
                log_Kv = log_K + log_v.unsqueeze(0)
                log_u = log_a - torch.logsumexp(log_Kv, dim=1)

                log_Ku = log_K + log_u.unsqueeze(1)
                log_v = log_b - torch.logsumexp(log_Ku, dim=0)

            log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
            P = F.softmax(log_P, dim=1)
        return P

    def compute_correlation_at_keypoints(self, feat_l, feat_r, keypoints):
        B, C, Hf, Wf = feat_l.shape
        patch_size = self.ext.patch
        N = keypoints.shape[1]

        disp_map = torch.zeros(B, N, device=feat_l.device)

        for b in range(B):
            kps = keypoints[b]
            raw_row = torch.round(kps[:, 1] / patch_size).long().clamp(0, Hf - 1)
            raw_col = torch.round(kps[:, 0] / patch_size).long().clamp(0, Wf - 1)

            valid_kp = (kps[:, 0] > 0) | (kps[:, 1] > 0)
            row_feat = raw_row[valid_kp]
            col_feat = raw_col[valid_kp]
            if len(row_feat) == 0:
                continue

            unique_rows = row_feat.unique()

            for row_idx in unique_rows:
                mask_row = row_feat == row_idx
                kp_indices_in_valid = mask_row.nonzero(as_tuple=True)[0]
                n_kp = len(kp_indices_in_valid)
                if n_kp == 0:
                    continue

                left_row = feat_l[b, :, row_idx, :]
                right_row = feat_r[b, :, row_idx, :]

                cols = col_feat[kp_indices_in_valid]
                left_desc = left_row[:, cols].T
                left_desc = F.normalize(left_desc, dim=-1)
                right_row_norm = F.normalize(right_row.T, dim=-1)

                corr = torch.mm(left_desc, right_row_norm.T)
                corr = corr * self.temperature.abs().clamp(min=0.1)

                corr_refined = self.corr_refine(corr)

                if n_kp == 1:
                    prob = F.softmax(corr_refined, dim=-1)
                else:
                    cost = -corr_refined
                    prob = self.sinkhorn(cost, self.sinkhorn_eps)

                col_range = torch.arange(Wf, device=feat_l.device).float()
                expected_col = (prob * col_range.unsqueeze(0)).sum(dim=-1)

                disp_feat = cols.float() - expected_col
                disp_pixel = disp_feat * patch_size

                kp_indices_original = valid_kp.nonzero(as_tuple=True)[0][kp_indices_in_valid]
                disp_map[b, kp_indices_original] = disp_pixel

        return disp_map

    def forward(self, lg, rg, lrgb, rrgb, mask, cached_data=None):
        if cached_data is not None:
            feat_l = cached_data['feat_left']
            feat_r = cached_data['feat_right']
            kpl = cached_data['keypoints_left']
            sl = cached_data['scores_left']
            kpr = cached_data['keypoints_right']
            sr = cached_data['scores_right']
        else:
            kpl, sl = self.det(lg, mask)
            kpr, sr = self.det(rg, torch.ones_like(rg))
            feat_l = self.ext.forward_dense(lrgb)
            feat_r = self.ext.forward_dense(rrgb)

        feat_l_proj = self.proj_l(feat_l)
        feat_r_proj = self.proj_r(feat_r)

        disparity = self.compute_correlation_at_keypoints(
            feat_l_proj, feat_r_proj, kpl
        )

        kp_right_x = kpl[:, :, 0] - disparity
        kp_right_pred = torch.stack([kp_right_x, kpl[:, :, 1]], dim=-1)

        return {
            'keypoints_left': kpl,
            'scores_left': sl,
            'keypoints_right': kpr,
            'scores_right': sr,
            'keypoints_right_pred': kp_right_pred,
            'disparity': disparity,
            'match_scores': sl.unsqueeze(-1),
        }
