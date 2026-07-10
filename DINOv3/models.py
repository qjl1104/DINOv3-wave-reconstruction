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
    DINOv3 + 几何指纹融合的立体匹配模型。

    核心思路：
      DINOv3 特征提供了全局上下文（但小圆片间区分度不足），
      几何指纹（KNN 相对位置）提供了小圆片"星座"的唯一性标识。
      两者拼接后通过 geo_fusion 层融合，让网络学习最优组合。

    Architecture:
      1. DINOv3 backbone → dense feature maps [B, C, Hf, Wf]
      2. Feature projection: C → D (learnable, shared)
      3. 几何指纹：对每个关键点计算其 K 近邻的相对位置 → [2K]
      4. 融合：Linear(D + 2K, D) → 融合特征
      5. 1D 相关体 + Sinkhorn 全局匹配
      6. Soft-argmax → 亚像素视差
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.det = SparseKeypointDetector(cfg)
        self.ext = DINOv3FeatureExtractor(cfg)

        feat_dim = self.ext.feat_dim
        proj_dim = cfg.CORR_PROJ_DIM
        geo_dim = cfg.GEO_KNN_K * 2  # 几何指纹维度 = K * 2 (x, y)

        # 左右图共享投影权重，保证左右描述子在同一特征空间
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim, 1),
        )

        # DINO + 几何指纹融合层（可学习，左右图共享）
        self.geo_fusion = nn.Sequential(
            nn.Linear(proj_dim + geo_dim, cfg.GEO_FUSION_DIM),
            nn.GELU(),
            nn.Linear(cfg.GEO_FUSION_DIM, cfg.GEO_FUSION_DIM),
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

    def compute_geo_fingerprint_at_positions(self, query_positions, all_keypoints, K):
        """
        为一组查询位置计算几何指纹（K 近邻的相对位置向量，按角度排序）。

        几何指纹天然区分不同位置的小圆片：
          - 不同位置的"星座"不同 → 指纹不同
          - 近邻位置 → 指纹相似（连续变化）
          - 远距离位置 → 指纹不同

        Args:
            query_positions: [Q, 2] 查询点的像素坐标
            all_keypoints:   [N, 2] 所有关键点（用于找邻居）
            K:               近邻数量

        Returns:
            fingerprints: [Q, 2*K] 每个查询点的几何指纹
        """
        Q = query_positions.shape[0]
        N = all_keypoints.shape[0]
        if N < K + 1 or Q == 0:
            return torch.zeros(Q, 2 * K, device=query_positions.device)

        # 计算所有查询点到所有关键点的距离 [Q, N]
        # 只用同一行做向量化：dist_{q,n} = sqrt((x_q - x_n)² + (y_q - y_n)²)
        diff = query_positions.unsqueeze(1) - all_keypoints.unsqueeze(0)  # [Q, N, 2]
        dists = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2).sqrt()  # [Q, N]

        # 取 K 近邻
        _, knn_idx = dists.topk(K, dim=-1, largest=False)  # [Q, K]

        # 提取相对位置向量
        neighbors = diff[torch.arange(Q).unsqueeze(1), knn_idx]  # [Q, K, 2]

        # 按角度排序（保证旋转不变性——旋转后所有邻居一起转，排序后指纹不变）
        angles = torch.atan2(neighbors[:, :, 1], neighbors[:, :, 0])  # [Q, K]
        sorted_idx = torch.argsort(angles, dim=-1)  # [Q, K]
        neighbors_sorted = neighbors[torch.arange(Q).unsqueeze(1), sorted_idx]  # [Q, K, 2]

        # 归一化到 [-1, 1]：除以图像对角线长度，与 DINO 特征处于同一量级
        # 避免梯度爆炸（原始像素坐标可达数百，远超 DINO 特征的 ~[-0.5, 0.5]）
        diag = (self.cfg.IMAGE_WIDTH ** 2 + self.cfg.IMAGE_HEIGHT ** 2) ** 0.5
        neighbors_sorted = neighbors_sorted / diag

        return neighbors_sorted.reshape(Q, -1)  # [Q, 2*K]

    def compute_geo_fingerprint_row(self, all_keypoints, row_idx, Wf, patch_size, K):
        """
        为右图某一行所有列位置计算几何指纹。

        左图用关键点位置（稀疏），右图需要对整行所有列位置（密集）都计算，
        因为相关体是 [n_kp, Wf] 的密集矩阵。

        Args:
            all_keypoints: [N, 2] 右图所有关键点
            row_idx:       特征图行索引
            Wf:            特征图宽度
            patch_size:    DINO patch 大小
            K:             近邻数量

        Returns:
            fingerprints: [Wf, 2*K] 该行所有列位置的几何指纹
        """
        # 该行所有列位置对应的像素坐标
        cols = torch.arange(Wf, device=all_keypoints.device).float() * patch_size + patch_size / 2
        row_y = torch.full((Wf,), row_idx * patch_size + patch_size / 2, device=all_keypoints.device)
        query_positions = torch.stack([cols, row_y], dim=-1)  # [Wf, 2]

        return self.compute_geo_fingerprint_at_positions(query_positions, all_keypoints, K)

    def compute_correlation_at_keypoints(self, feat_l, feat_r, keypoints_l, keypoints_r):
        """
        计算相关体：DINO 特征 + 几何指纹融合后做内积。

        相比纯 DINO 版本，这里：
          1. 对左图关键点计算几何指纹 [n_kp, 2K]
          2. 对右图该行所有列计算几何指纹 [Wf, 2K]  
          3. 与 DINO 特征拼接后通过 geo_fusion 融合
          4. 在融合特征空间计算相关体
        """
        B, C, Hf, Wf = feat_l.shape
        patch_size = self.ext.patch
        N = keypoints_l.shape[1]
        K = self.cfg.GEO_KNN_K

        disp_map = torch.zeros(B, N, device=feat_l.device)
        prob_list = []

        for b in range(B):
            kps_l = keypoints_l[b]
            kps_r = keypoints_r[b]

            raw_row = torch.round(kps_l[:, 1] / patch_size).long().clamp(0, Hf - 1)
            raw_col = torch.round(kps_l[:, 0] / patch_size).long().clamp(0, Wf - 1)

            valid_kp = (kps_l[:, 0] > 0) | (kps_l[:, 1] > 0)
            row_feat = raw_row[valid_kp]
            col_feat = raw_col[valid_kp]
            if len(row_feat) == 0:
                continue

            # 右图有效关键点（用于几何指纹计算）
            valid_kp_r = (kps_r[:, 0] > 0) | (kps_r[:, 1] > 0)
            kps_r_valid = kps_r[valid_kp_r]

            unique_rows = row_feat.unique()

            for row_idx in unique_rows:
                mask_row = row_feat == row_idx
                kp_indices_in_valid = mask_row.nonzero(as_tuple=True)[0]
                n_kp = len(kp_indices_in_valid)
                if n_kp == 0:
                    continue

                # --- DINO 特征 ---
                left_row = feat_l[b, :, row_idx, :]    # [C, Wf]
                right_row = feat_r[b, :, row_idx, :]   # [C, Wf]

                cols = col_feat[kp_indices_in_valid]
                left_desc_dino = left_row[:, cols].T   # [n_kp, D]

                # --- 几何指纹 ---
                # 左图：该行关键点位置的几何指纹
                query_positions_l = kps_l[valid_kp][kp_indices_in_valid]  # [n_kp, 2]
                geo_fp_l = self.compute_geo_fingerprint_at_positions(
                    query_positions_l, kps_l[valid_kp], K
                )  # [n_kp, 2K]

                # 右图：该行所有列位置的几何指纹
                geo_fp_r = self.compute_geo_fingerprint_row(
                    kps_r_valid, row_idx, Wf, patch_size, K
                )  # [Wf, 2K]

                # --- DINO + 几何融合 ---
                left_desc_fused = self.geo_fusion(
                    torch.cat([left_desc_dino, geo_fp_l], dim=-1)
                )  # [n_kp, D_fusion]

                right_desc_fused = self.geo_fusion(
                    torch.cat([right_row.T, geo_fp_r], dim=-1)
                )  # [Wf, D_fusion]

                # --- 相关体 ---
                left_desc = F.normalize(left_desc_fused, dim=-1)
                right_row_norm = F.normalize(right_desc_fused, dim=-1)

                corr = torch.mm(left_desc, right_row_norm.T)
                corr = corr * self.temperature.abs().clamp(min=0.1)

                corr_refined = self.corr_refine(corr)

                if n_kp == 1:
                    prob = F.softmax(corr_refined, dim=-1)
                else:
                    cost = -corr_refined
                    prob = self.sinkhorn(cost, self.sinkhorn_eps)

                prob_list.append(prob)

                col_range = torch.arange(Wf, device=feat_l.device).float()
                expected_col = (prob * col_range.unsqueeze(0)).sum(dim=-1)

                disp_feat = cols.float() - expected_col
                disp_pixel = disp_feat * patch_size

                kp_indices_original = valid_kp.nonzero(as_tuple=True)[0][kp_indices_in_valid]
                disp_map[b, kp_indices_original] = disp_pixel

        return disp_map, prob_list

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

        feat_l_proj = self.proj(feat_l)
        feat_r_proj = self.proj(feat_r)

        disparity, prob_list = self.compute_correlation_at_keypoints(
            feat_l_proj, feat_r_proj, kpl, kpr
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
            'correlation_probs': prob_list,  # Sinkhorn 软分配矩阵列表
        }
