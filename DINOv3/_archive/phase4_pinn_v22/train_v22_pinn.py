# train_v22_pinn.py
# [V22.0 策略A - 强物理约束 PINN 版]
# ----------------------------------------------------------------------
# 核心升级:
# 1. [PINN Loss] 引入三项物理约束：局部平滑、最大波陡限制、零均值重力约束。
# 2. [单位归一化] 强制将坐标转为米 (m) 计算 Loss，激活 SmoothL1 的微调能力。
# 3. [DINO 修复] 采用倒序切片，兼容 DINOv2/v3 的 Register Tokens。
# ----------------------------------------------------------------------

import os
import sys
import glob
import json
from datetime import datetime
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Tensorboard not found. Skipping tensorboard logging.")
    SummaryWriter = None

try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80 + "\n[FATAL] transformers not found. pip install transformers\n" + "=" * 80)
    sys.exit(1)

# --- 路径配置 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


@dataclass
class Config:
    """训练配置"""
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")

    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs_strategy_A")
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")

    IMAGE_HEIGHT: int = 0
    IMAGE_WIDTH: int = 0
    MASK_THRESHOLD: int = 30

    MAX_KEYPOINTS: int = 1024
    BLOB_MIN_THRESHOLD: float = 15.0
    BLOB_MIN_AREA: float = 10.0
    BLOB_MAX_AREA: float = 2500.0

    FEATURE_DIM: int = 768
    NUM_ATTENTION_LAYERS: int = 6
    NUM_HEADS: int = 8
    MATCHING_TEMPERATURE: float = 15.0

    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 1e-5
    NUM_EPOCHS: int = 150

    # [Strategy A 权重配置]
    # Photo Loss: 1.0 (基础匹配)
    # Physics - Smoothness: 1.0 (局部平滑)
    # Physics - Slope: 0.5 (波陡限制，防止尖刺)
    # Physics - ZeroMean: 0.2 (防止整体漂移)
    PHOTOMETRIC_WEIGHT: float = 1.0
    PHY_SMOOTH_WEIGHT: float = 1.0
    PHY_SLOPE_WEIGHT: float = 0.5
    PHY_ZEROMEAN_WEIGHT: float = 0.2

    PATCH_SIZE_PHOTOMETRIC: int = 11
    VISUALIZE_INTERVAL: int = 10


# --- 核心组件: PINN 物理损失函数 ---
class PINNPhysicsLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.PATCH_SIZE_PHOTOMETRIC

    def disparity_to_3d(self, keypoints, disparity, Q):
        """将视差转换为 3D 坐标 (输出单位 mm)"""
        B, N, _ = keypoints.shape
        device = keypoints.device
        disp_unsqueezed = disparity.unsqueeze(-1)
        ones = torch.ones((B, N, 1), device=device)
        points_4d = torch.cat([keypoints, disp_unsqueezed, ones], dim=-1)
        projected = torch.bmm(points_4d, Q.transpose(1, 2))
        X, Y, Z, W = projected.unbind(-1)
        W = torch.clamp(W, min=1e-6)
        points_3d = torch.stack([X / W, Y / W, Z / W], dim=-1)
        return points_3d

    def compute_pinn_loss(self, points_3d, scores):
        """
        策略 A 的核心：多维度物理约束
        """
        B, N, _ = points_3d.shape
        if N < 10:
            return (torch.tensor(0.0).to(points_3d.device),
                    torch.tensor(0.0).to(points_3d.device),
                    torch.tensor(0.0).to(points_3d.device))

        loss_smooth = 0.0
        loss_slope = 0.0
        loss_zeromean = 0.0
        valid_batches = 0

        for b in range(B):
            p3d = points_3d[b]
            sc = scores[b]

            # 1. 空间过滤 (mm)
            valid_mask = (p3d[:, 2] > 500) & (p3d[:, 2] < 15000) & (sc > 0.1)
            p_valid = p3d[valid_mask]

            if p_valid.shape[0] < 10: continue

            # [关键] 归一化到米 (m)，解决数值量级问题
            z_m = p_valid[:, 2] / 1000.0
            xy_m = p_valid[:, :2] / 1000.0

            # 2. KNN 寻找邻居
            dist_matrix = torch.cdist(xy_m, xy_m)
            K = 5
            if p_valid.shape[0] <= K: K = p_valid.shape[0] - 1

            # values: (M, K), indices: (M, K)
            dists, indices = dist_matrix.topk(K + 1, largest=False, dim=1)
            neighbor_indices = indices[:, 1:]
            neighbor_dists = dists[:, 1:]  # 邻居距离 (米)

            neighbor_z_m = z_m[neighbor_indices]

            # --- 约束 1: 局部平滑 (Laplacian) ---
            # 目标：Z 值不应偏离邻居均值太远
            local_mean_z = neighbor_z_m.mean(dim=1)
            # beta=0.05 表示允许 5cm 内的线性误差，超过后平方惩罚
            l_smooth = F.smooth_l1_loss(z_m, local_mean_z, beta=0.05)

            # --- 约束 2: 波陡约束 (Slope Constraint) ---
            # 目标：|dz/dr| < max_slope
            delta_z = (neighbor_z_m - z_m.unsqueeze(1)).abs()
            delta_xy = torch.clamp(neighbor_dists, min=1e-3)  # 防止除零

            slopes = delta_z / delta_xy  # 计算局部斜率

            # 物理先验：规则波斜率通常 < 0.15。惩罚超过 0.3 的异常斜率(飞点)
            slope_penalty = F.relu(slopes - 0.3)
            l_slope = slope_penalty.mean()

            # --- 约束 3: 零均值/范围约束 (Global Consistency) ---
            # 目标：防止整体 Z 值漂移
            median_z = z_m.median()
            # 惩罚偏离中位数超过 0.2m (200mm) 的点
            range_penalty = F.relu((z_m - median_z).abs() - 0.2)
            l_zeromean = range_penalty.mean()

            loss_smooth += l_smooth
            loss_slope += l_slope
            loss_zeromean += l_zeromean
            valid_batches += 1

        if valid_batches == 0:
            return (torch.tensor(0.0).to(points_3d.device),
                    torch.tensor(0.0).to(points_3d.device),
                    torch.tensor(0.0).to(points_3d.device))

        return (loss_smooth / valid_batches,
                loss_slope / valid_batches,
                loss_zeromean / valid_batches)

    def sample_patches(self, image, keypoints, patch_size):
        B, N, _ = keypoints.shape
        B, C, H, W = image.shape
        device = keypoints.device
        half = patch_size // 2
        xv, yv = torch.meshgrid(torch.linspace(-half, half, patch_size),
                                torch.linspace(-half, half, patch_size), indexing='ij')
        grid_rel = torch.stack([yv, xv], dim=-1).to(device).view(1, 1, -1, 2)
        kp_expand = keypoints.unsqueeze(2) + grid_rel
        kp_norm = kp_expand.clone()
        kp_norm[..., 0] = 2 * kp_norm[..., 0] / (W - 1) - 1
        kp_norm[..., 1] = 2 * kp_norm[..., 1] / (H - 1) - 1
        kp_norm = kp_norm.view(B, -1, 1, 2)
        patches = F.grid_sample(image, kp_norm, align_corners=True, mode='bilinear', padding_mode='border')
        patches = patches.view(B, C, N, patch_size, patch_size)
        return patches

    def forward(self, left_gray, right_gray, keypoints_left, disparity, scores_left, Q):
        # 1. 光度损失
        kp_right_pred = keypoints_left.clone()
        kp_right_pred[..., 0] = kp_right_pred[..., 0] - disparity

        patches_l = self.sample_patches(left_gray, keypoints_left, self.patch_size)
        patches_r = self.sample_patches(right_gray, kp_right_pred, self.patch_size)

        loss_pixel = F.l1_loss(patches_l, patches_r, reduction='none')
        photo_loss_per_kp = loss_pixel.mean(dim=[1, 3, 4])

        mask = (scores_left > 0.1) & (disparity > 0.1)
        if mask.sum() > 0:
            photo_loss = (photo_loss_per_kp * mask).sum() / mask.sum()
        else:
            photo_loss = torch.tensor(0.0, device=left_gray.device)

        # 2. PINN 物理损失
        points_3d = self.disparity_to_3d(keypoints_left, disparity, Q)
        l_smooth, l_slope, l_zeromean = self.compute_pinn_loss(points_3d, scores_left)

        return photo_loss, l_smooth, l_slope, l_zeromean


# --- 数据集 ---
class RectifiedWaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images: print(f"[Warning] No images in {cfg.LEFT_IMAGE_DIR}")
        calib = np.load(cfg.CALIBRATION_FILE)
        self.map1_l, self.map2_l = calib['map1_left'], calib['map2_left']
        self.map1_r, self.map2_r = calib['map1_right'], calib['map2_right']
        self.roi_l = tuple(map(int, calib['roi_left']))
        self.roi_r = tuple(map(int, calib['roi_right']))
        self.Q_base = calib['Q']
        w_l, h_l = self.roi_l[2], self.roi_l[3]
        w_r, h_r = self.roi_r[2], self.roi_r[3]
        self.target_w = min(w_l, w_r)
        self.target_h = min(h_l, h_r)
        if cfg.IMAGE_WIDTH == 0:
            cfg.IMAGE_WIDTH = self.target_w
            cfg.IMAGE_HEIGHT = self.target_h
        indices = np.arange(len(self.left_images))
        split = int(len(indices) * 0.9)
        self.indices = indices[split:] if is_validation else indices[:split]

    def __len__(self):
        return len(self.indices)

    def get_corrected_Q(self):
        Q = self.Q_base.copy()
        Q[0, 3] += self.roi_l[0];
        Q[1, 3] += self.roi_l[1]
        return torch.from_numpy(Q).float()

    def __getitem__(self, idx):
        idx = self.indices[idx]
        l_path = self.left_images[idx]
        filename = os.path.basename(l_path)
        if "left" in filename:
            r_name = filename.replace("left", "right")
        elif "Left" in filename:
            r_name = filename.replace("Left", "Right")
        else:
            r_name = "right" + filename[4:]
        r_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, r_name)
        l_raw = cv2.imread(l_path, 0)
        r_raw = cv2.imread(r_path, 0)
        if l_raw is None or r_raw is None: return None
        l_rect = cv2.remap(l_raw, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        r_rect = cv2.remap(r_raw, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        x, y, w, h = self.roi_l
        l_crop = l_rect[y:y + h, x:x + w][:self.target_h, :self.target_w]
        x, y, w, h = self.roi_r
        r_crop = r_rect[y:y + h, x:x + w][:self.target_h, :self.target_w]
        _, mask = cv2.threshold(l_crop, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        l_tensor = torch.from_numpy(l_crop).float().unsqueeze(0) / 255.0
        r_tensor = torch.from_numpy(r_crop).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        l_rgb = cv2.cvtColor(l_crop, cv2.COLOR_GRAY2RGB)
        r_rgb = cv2.cvtColor(r_crop, cv2.COLOR_GRAY2RGB)
        l_rgb_t = torch.from_numpy(l_rgb.transpose(2, 0, 1)).float() / 255.0
        r_rgb_t = torch.from_numpy(r_rgb.transpose(2, 0, 1)).float() / 255.0
        Q = self.get_corrected_Q()
        return {'left_gray': l_tensor, 'right_gray': r_tensor, 'left_rgb': l_rgb_t, 'right_rgb': r_rgb_t,
                'mask': mask_tensor, 'Q': Q}


# --- 模型定义 ---
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
        p.filterByCircularity = True
        p.minCircularity = 0.1
        p.filterByConvexity = True
        p.minConvexity = 0.85
        p.filterByInertia = True
        p.minInertiaRatio = 0.1
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
        max_l = max([len(k) for k in kpts])
        k_pad, s_pad = [], []
        for k, s in zip(kpts, scores):
            pad_n = max_l - len(k)
            if pad_n > 0:
                k = torch.cat([k, torch.zeros(pad_n, 2, device=img.device)], 0)
                s = torch.cat([s, torch.zeros(pad_n, device=img.device)], 0)
            k_pad.append(k)
            s_pad.append(s)
        return torch.stack(k_pad), torch.stack(s_pad)


class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dino = AutoModel.from_pretrained(cfg.DINO_LOCAL_PATH, local_files_only=True)
        for p in self.dino.parameters(): p.requires_grad = False
        self.feat_dim = self.dino.config.hidden_size
        self.patch = self.dino.config.patch_size

    def forward(self, img, kpts):
        with torch.no_grad():
            out = self.dino(img).last_hidden_state
        B, _, H, W = img.shape
        n_patches_h = H // self.patch
        n_patches_w = W // self.patch
        # 倒序切片，确保兼容性
        feat = out[:, -(n_patches_h * n_patches_w):]
        feat = feat.transpose(1, 2).reshape(B, self.feat_dim, n_patches_h, n_patches_w)
        grid = kpts.clone()
        grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
        grid = grid.unsqueeze(2)
        desc = F.grid_sample(feat, grid, align_corners=True, padding_mode='border')
        return desc.squeeze(3).transpose(1, 2)


class SparseMatchingStereoModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.det = SparseKeypointDetector(cfg)
        self.ext = DINOv3FeatureExtractor(cfg)
        self.proj = nn.Linear(2, cfg.FEATURE_DIM)
        layer = nn.TransformerEncoderLayer(d_model=cfg.FEATURE_DIM, nhead=cfg.NUM_HEADS, batch_first=True)
        self.trans = nn.TransformerEncoder(layer, num_layers=cfg.NUM_ATTENTION_LAYERS)
        self.out_proj = nn.Linear(cfg.FEATURE_DIM, cfg.FEATURE_DIM)

    def forward(self, lg, rg, lrgb, rrgb, mask):
        kpl, sl = self.det(lg, mask)
        kpr, sr = self.det(rg, torch.ones_like(rg))
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
        mask_geo = (kpl[:, :, 1].unsqueeze(2) - kpr[:, :, 1].unsqueeze(1)).abs() < 2.0
        scores = scores.masked_fill(~mask_geo, -1e9)
        probs = F.softmax(scores, dim=-1)
        x_right_ex = (probs * kpr[:, :, 0].unsqueeze(1)).sum(dim=2)
        disp = kpl[:, :, 0] - x_right_ex
        return {'keypoints_left': kpl, 'scores_left': sl, 'disparity': disp}


# --- 训练器 (Strategy A 专用) ---
class StrategyATrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR, self.timestamp)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.vis_dir = os.path.join(self.run_dir, "vis")

        for d in [self.ckpt_dir, self.log_dir, self.vis_dir]:
            os.makedirs(d, exist_ok=True)

        self.model = SparseMatchingStereoModel(cfg).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE)
        self.loss_fn = PINNPhysicsLoss(cfg)

        ds = RectifiedWaveStereoDataset(cfg)
        self.loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=self.collate)

        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tensorboard")) if SummaryWriter else None
        self.log_file = os.path.join(self.log_dir, "training_log.json")
        self.history = {'total_loss': [], 'photo_loss': [], 'smooth_loss': [], 'slope_loss': [], 'mean_loss': []}

        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            json.dump(asdict(cfg), f, indent=2)

    @staticmethod
    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        return torch.utils.data.dataloader.default_collate(batch)

    def update_json_log(self, epoch):
        data = {
            'epoch': epoch,
            'history': self.history,
            'config': asdict(self.cfg)
        }
        try:
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write JSON log: {e}")

    def plot_history(self):
        epochs = range(1, len(self.history['total_loss']) + 1)
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history['total_loss'], label='Total', color='black')
        plt.title('Total Loss')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['photo_loss'], label='Photo', color='blue')
        plt.plot(epochs, self.history['smooth_loss'], label='Smoothness', color='green')
        plt.title('Matching & Smoothness')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history['slope_loss'], label='Slope (Steepness)', color='red')
        plt.plot(epochs, self.history['mean_loss'], label='Zero-Mean', color='orange')
        plt.title('Physics Constraints')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "loss_history_strategyA.png"))
        plt.close()

    def train(self):
        print(f"--- Strategy A Training (PINN Mode) ---")
        print(f"Weights -> Photo: {self.cfg.PHOTOMETRIC_WEIGHT}, Smooth: {self.cfg.PHY_SMOOTH_WEIGHT}")
        print(f"           Slope: {self.cfg.PHY_SLOPE_WEIGHT}, ZeroMean: {self.cfg.PHY_ZEROMEAN_WEIGHT}")

        for epoch in range(self.cfg.NUM_EPOCHS):
            ep_stats = {'total': 0, 'photo': 0, 'smooth': 0, 'slope': 0, 'mean': 0}
            count = 0

            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}")
            for batch in pbar:
                if batch is None: continue

                lg = batch['left_gray'].to(self.device)
                rg = batch['right_gray'].to(self.device)
                lrgb = batch['left_rgb'].to(self.device)
                rrgb = batch['right_rgb'].to(self.device)
                mask = batch['mask'].to(self.device)
                Q = batch['Q'].to(self.device)

                h, w = lg.shape[2:]
                padh, padw = (14 - h % 14) % 14, (14 - w % 14) % 14
                if padh > 0 or padw > 0:
                    lg = F.pad(lg, (0, padw, 0, padh))
                    rg = F.pad(rg, (0, padw, 0, padh))
                    lrgb = F.pad(lrgb, (0, padw, 0, padh))
                    rrgb = F.pad(rrgb, (0, padw, 0, padh))
                    mask = F.pad(mask, (0, padw, 0, padh))

                self.optimizer.zero_grad()
                out = self.model(lg, rg, lrgb, rrgb, mask)

                l_photo, l_smooth, l_slope, l_zeromean = self.loss_fn(
                    lg, rg, out['keypoints_left'], out['disparity'], out['scores_left'], Q
                )

                w_photo = self.cfg.PHOTOMETRIC_WEIGHT * l_photo
                w_smooth = self.cfg.PHY_SMOOTH_WEIGHT * l_smooth
                w_slope = self.cfg.PHY_SLOPE_WEIGHT * l_slope
                w_zeromean = self.cfg.PHY_ZEROMEAN_WEIGHT * l_zeromean

                loss = w_photo + w_smooth + w_slope + w_zeromean

                loss.backward()
                self.optimizer.step()

                ep_stats['total'] += loss.item()
                ep_stats['photo'] += w_photo.item()
                ep_stats['smooth'] += w_smooth.item()
                ep_stats['slope'] += w_slope.item()
                ep_stats['mean'] += w_zeromean.item()
                count += 1

                pbar.set_postfix({'L': loss.item(), 'Slope': l_slope.item()})

            if count > 0:
                self.history['total_loss'].append(ep_stats['total'] / count)
                self.history['photo_loss'].append(ep_stats['photo'] / count)
                self.history['smooth_loss'].append(ep_stats['smooth'] / count)
                self.history['slope_loss'].append(ep_stats['slope'] / count)
                self.history['mean_loss'].append(ep_stats['mean'] / count)

            self.update_json_log(epoch)
            if (epoch + 1) % self.cfg.VISUALIZE_INTERVAL == 0:
                self.plot_history()

            if (epoch + 1) % 10 == 0:
                path = os.path.join(self.ckpt_dir, f"model_ep{epoch + 1}.pth")
                torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    cfg = Config()
    trainer = StrategyATrainer(cfg)
    trainer.train()