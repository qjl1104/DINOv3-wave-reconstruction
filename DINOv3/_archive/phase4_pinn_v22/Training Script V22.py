# train_v22_pure_full_resolution_fixed.py
# [V22.10 策略A - 绝对路径修复版]
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


# --- 路径检查函数 ---
def check_path(path, name):
    if not os.path.exists(path):
        print(f"\n{'=' * 40}")
        print(f"[严重错误] 找不到 {name}！")
        print(f"试图寻找的路径是: {path}")
        print(f"{'=' * 40}\n")
        sys.exit(1)
    else:
        print(f"[检查通过] 找到 {name}: {path}")


# --- 配置区域 ---
@dataclass
class Config:
    """训练配置"""

    # 1. 图片文件夹 (请确保路径正确)
    LEFT_IMAGE_DIR: str = r"D:\Research\wave_reconstruction_project\data\left_images"
    RIGHT_IMAGE_DIR: str = r"D:\Research\wave_reconstruction_project\data\right_images"

    # 2. 标定文件 (硬编码为您提供的绝对路径)
    CALIBRATION_FILE: str = r"D:\Research\wave_reconstruction_project\DINOv3\1125\calibration_paper_params.npz"

    # 3. DINO模型路径
    DINO_LOCAL_PATH: str = "dinov3-base-model"

    # 其他配置保持不变
    RUNS_BASE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_runs_strategy_A")
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
    ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 1e-5
    NUM_EPOCHS: int = 150
    PHOTOMETRIC_WEIGHT: float = 1.0
    EPIPOLAR_WEIGHT: float = 0.1
    PHY_SMOOTH_WEIGHT: float = 20.0
    PHY_SLOPE_WEIGHT: float = 10.0
    PHY_ZEROMEAN_WEIGHT: float = 5.0
    PATCH_SIZE_PHOTOMETRIC: int = 11
    VISUALIZE_INTERVAL: int = 5


# --- PINN 物理损失函数 ---
class PINNPhysicsLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.PATCH_SIZE_PHOTOMETRIC

    def disparity_to_3d(self, keypoints, disparity, Q):
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
        B, N, _ = points_3d.shape
        device = points_3d.device
        if N < 10: return (torch.tensor(0.0, device=device),) * 3

        loss_smooth = 0.0
        loss_slope = 0.0
        loss_zeromean = 0.0
        valid_batches = 0

        for b in range(B):
            p3d = points_3d[b]
            sc = scores[b]
            valid_mask = (p3d[:, 2] > 500) & (p3d[:, 2] < 30000) & (sc > 0.1)
            p_valid = p3d[valid_mask]
            if p_valid.shape[0] < 10: continue

            p_valid_m = p_valid / 1000.0
            xy_m = p_valid_m[:, [0, 2]]
            z_height_m = p_valid_m[:, 1]

            if p_valid_m.shape[0] > 2000:
                perm = torch.randperm(p_valid_m.shape[0])[:2000]
                xy_m = xy_m[perm]
                z_height_m = z_height_m[perm]

            dist_matrix = torch.cdist(xy_m, xy_m)
            K = 5
            if xy_m.shape[0] <= K: K = xy_m.shape[0] - 1

            dists, indices = dist_matrix.topk(K + 1, largest=False, dim=1)
            neighbor_indices = indices[:, 1:]
            neighbor_dists = torch.clamp(dists[:, 1:], min=0.001)
            neighbor_heights = z_height_m[neighbor_indices]

            local_mean = neighbor_heights.mean(dim=1)
            l_smooth = F.smooth_l1_loss(z_height_m, local_mean, beta=0.01)

            delta_h = (neighbor_heights - z_height_m.unsqueeze(1)).abs()
            slopes = delta_h / neighbor_dists
            slope_penalty = F.relu(slopes - 0.4)
            l_slope = slope_penalty.mean()

            mean_height = z_height_m.mean()
            l_zeromean = mean_height.abs()

            loss_smooth += l_smooth
            loss_slope += l_slope
            loss_zeromean += l_zeromean
            valid_batches += 1

        if valid_batches == 0: return (torch.tensor(0.0, device=device),) * 3
        return (loss_smooth / valid_batches, loss_slope / valid_batches, loss_zeromean / valid_batches)

    def sample_patches(self, image, keypoints, patch_size):
        B, N, _ = keypoints.shape
        device = keypoints.device
        half = patch_size // 2
        xv, yv = torch.meshgrid(torch.linspace(-half, half, patch_size),
                                torch.linspace(-half, half, patch_size), indexing='ij')
        grid_rel = torch.stack([yv, xv], dim=-1).to(device).view(1, 1, -1, 2)
        kp_expand = keypoints.unsqueeze(2) + grid_rel
        b, c, h, w = image.shape
        kp_norm = kp_expand.clone()
        kp_norm[..., 0] = 2 * kp_norm[..., 0] / (w - 1) - 1
        kp_norm[..., 1] = 2 * kp_norm[..., 1] / (h - 1) - 1
        kp_norm = kp_norm.view(B, -1, 1, 2)
        patches = F.grid_sample(image, kp_norm, align_corners=True, mode='bilinear', padding_mode='border')
        patches = patches.view(B, c, N, patch_size, patch_size)
        return patches

    def forward(self, left_gray, right_gray, keypoints_left, keypoints_right, scores, Q):
        y_diff = (keypoints_left[..., 1] - keypoints_right[..., 1]).abs()
        mask_conf = scores > 0.1
        if mask_conf.sum() > 0:
            l_epipolar = F.smooth_l1_loss(y_diff[mask_conf], torch.zeros_like(y_diff[mask_conf]), beta=1.0)
        else:
            l_epipolar = torch.tensor(0.0, device=left_gray.device)

        patches_l = self.sample_patches(left_gray, keypoints_left, self.patch_size)
        patches_r = self.sample_patches(right_gray, keypoints_right, self.patch_size)
        loss_pixel = F.l1_loss(patches_l, patches_r, reduction='none').mean(dim=[1, 3, 4])

        if mask_conf.sum() > 0:
            l_photo = (loss_pixel * mask_conf).sum() / mask_conf.sum()
        else:
            l_photo = torch.tensor(0.0, device=left_gray.device)

        disparity = keypoints_left[..., 0] - keypoints_right[..., 0]
        valid_disp = disparity > 0.1
        mask_final = mask_conf & valid_disp

        if mask_final.sum() < 10:
            l_smooth, l_slope, l_zeromean = (torch.tensor(0.0, device=left_gray.device),) * 3
        else:
            points_3d = self.disparity_to_3d(keypoints_left, disparity, Q)
            l_smooth, l_slope, l_zeromean = self.compute_pinn_loss(points_3d, scores)

        return l_photo, l_epipolar, l_smooth, l_slope, l_zeromean


# --- 数据集 ---
class RectifiedWaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        # 再次检查路径
        check_path(cfg.CALIBRATION_FILE, "标定文件")
        check_path(cfg.LEFT_IMAGE_DIR, "左图文件夹")

        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            print(f"[错误] {cfg.LEFT_IMAGE_DIR} 下没有找到图片！")
            sys.exit(1)

        calib = np.load(cfg.CALIBRATION_FILE)
        self.map1_l = calib['map1_left']
        self.map2_l = calib['map2_left']
        self.map1_r = calib['map1_right']
        self.map2_r = calib['map2_right']
        self.Q_base = calib['Q']

        if 'roi_left' in calib:
            self.roi_l = tuple(map(int, calib['roi_left']))
            self.roi_r = tuple(map(int, calib['roi_right']))
        else:
            h, w = self.map1_l.shape[:2]
            self.roi_l = (0, 0, w, h)
            self.roi_r = (0, 0, w, h)

        if self.roi_l[2] <= 0:
            h, w = self.map1_l.shape[:2]
            self.roi_l = (0, 0, w, h)
            self.roi_r = (0, 0, w, h)

        self.orig_w = min(self.roi_l[2], self.roi_r[2])
        self.orig_h = min(self.roi_l[3], self.roi_r[3])

        if cfg.IMAGE_WIDTH == 0:
            cfg.IMAGE_WIDTH = self.orig_w
            cfg.IMAGE_HEIGHT = self.orig_h

        indices = np.arange(len(self.left_images))
        split = int(len(indices) * 0.9)
        self.indices = indices[split:] if is_validation else indices[:split]

    def __len__(self):
        return len(self.indices)

    def get_Q_tensor(self):
        return torch.from_numpy(self.Q_base).float()

    def __getitem__(self, idx):
        idx = self.indices[idx]
        l_path = self.left_images[idx]
        filename = os.path.basename(l_path)
        if "left" in filename:
            r_name = filename.replace("left", "right")
        elif "Left" in filename:
            r_name = filename.replace("Left", "Right")
        else:
            r_name = filename
        r_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, r_name)
        if not os.path.exists(r_path): return None

        l_raw = cv2.imread(l_path, 0)
        r_raw = cv2.imread(r_path, 0)
        if l_raw is None or r_raw is None: return None

        l_rect = cv2.remap(l_raw, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        r_rect = cv2.remap(r_raw, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        x_roi, y_roi, w_roi, h_roi = self.roi_l
        w_roi = min(w_roi, l_rect.shape[1] - x_roi)
        h_roi = min(h_roi, l_rect.shape[0] - y_roi)

        l_crop = l_rect[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]
        r_crop = r_rect[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]

        _, mask = cv2.threshold(l_crop, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        l_tensor = torch.from_numpy(l_crop).float().unsqueeze(0) / 255.0
        r_tensor = torch.from_numpy(r_crop).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        l_rgb = cv2.cvtColor(l_crop, cv2.COLOR_GRAY2RGB)
        r_rgb = cv2.cvtColor(r_crop, cv2.COLOR_GRAY2RGB)
        l_rgb_t = torch.from_numpy(l_rgb.transpose(2, 0, 1)).float() / 255.0
        r_rgb_t = torch.from_numpy(r_rgb.transpose(2, 0, 1)).float() / 255.0
        Q = self.get_Q_tensor()
        return {'left_gray': l_tensor, 'right_gray': r_tensor, 'left_rgb': l_rgb_t, 'right_rgb': r_rgb_t,
                'mask': mask_tensor, 'Q': Q}


# --- 模型定义 (无变化) ---
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
        max_l = max([len(k) for k in kpts])
        if max_l == 0: max_l = 1
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
        try:
            self.dino = AutoModel.from_pretrained(cfg.DINO_LOCAL_PATH, local_files_only=True)
        except:
            self.dino = AutoModel.from_pretrained("facebook/dinov2-base")
        for p in self.dino.parameters(): p.requires_grad = False
        self.feat_dim = self.dino.config.hidden_size
        self.patch = self.dino.config.patch_size

    def forward(self, img, kpts):
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
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
        probs = F.softmax(scores, dim=-1)
        x_right_ex = (probs * kpr[:, :, 0].unsqueeze(1)).sum(dim=2)
        y_right_ex = (probs * kpr[:, :, 1].unsqueeze(1)).sum(dim=2)
        kp_right_pred = torch.stack([x_right_ex, y_right_ex], dim=-1)
        return {'keypoints_left': kpl, 'scores_left': sl, 'keypoints_right_pred': kp_right_pred}


# --- 训练器 ---
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
        self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        ds = RectifiedWaveStereoDataset(cfg)
        self.loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=self.collate, num_workers=0)
        self.history = {'total_loss': [], 'photo_loss': [], 'epi_loss': [], 'smooth_loss': [], 'slope_loss': [],
                        'mean_loss': []}
        self.log_file = os.path.join(self.log_dir, "training_log.json")
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            json.dump(asdict(cfg), f, indent=2)

    @staticmethod
    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        return torch.utils.data.dataloader.default_collate(batch)

    def update_json_log(self, epoch):
        data = {'epoch': epoch, 'history': self.history, 'config': asdict(self.cfg)}
        try:
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write JSON log: {e}")

    def plot_history(self):
        epochs = range(1, len(self.history['total_loss']) + 1)
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history['total_loss'], label='Total', color='black')
        plt.title('Total Loss');
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['photo_loss'], label='Photo', color='blue')
        plt.plot(epochs, self.history['epi_loss'], label='Epipolar', color='purple')
        plt.title('Geometric');
        plt.legend();
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history['smooth_loss'], label='Smooth', color='green')
        plt.plot(epochs, self.history['slope_loss'], label='Slope', color='red')
        plt.title('Physics');
        plt.legend();
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "loss_history_detailed.png"))
        plt.close()

    def train(self):
        print(f"--- 开始训练 (V22.10 Pure Full Res + Path Check) ---")
        print(f"Device: {self.device}")
        print(f"标定文件: {self.cfg.CALIBRATION_FILE}")
        print(f"Batch Size: {self.cfg.BATCH_SIZE} (Accumulated to {self.cfg.BATCH_SIZE * self.cfg.ACCUMULATION_STEPS})")

        for epoch in range(self.cfg.NUM_EPOCHS):
            ep_stats = {'total': 0, 'photo': 0, 'epi': 0, 'smooth': 0, 'slope': 0, 'mean': 0}
            count = 0
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS}")
            self.optimizer.zero_grad()

            for step, batch in enumerate(pbar):
                if batch is None: continue
                lg = batch['left_gray'].to(self.device)
                rg = batch['right_gray'].to(self.device)
                lrgb = batch['left_rgb'].to(self.device)
                rrgb = batch['right_rgb'].to(self.device)
                mask = batch['mask'].to(self.device)
                Q = batch['Q'].to(self.device)

                h, w = lg.shape[2:]
                padh = (14 - h % 14) % 14
                padw = (14 - w % 14) % 14
                if padh > 0 or padw > 0:
                    lg = F.pad(lg, (0, padw, 0, padh))
                    rg = F.pad(rg, (0, padw, 0, padh))
                    lrgb = F.pad(lrgb, (0, padw, 0, padh))
                    rrgb = F.pad(rrgb, (0, padw, 0, padh))
                    mask = F.pad(mask, (0, padw, 0, padh))

                with torch.amp.autocast('cuda', enabled=True):
                    out = self.model(lg, rg, lrgb, rrgb, mask)
                    kpl = out['keypoints_left']
                    kpr_pred = out['keypoints_right_pred']
                    scores = out['scores_left']
                    l_photo, l_epi, l_smooth, l_slope, l_zeromean = self.loss_fn(lg, rg, kpl, kpr_pred, scores, Q)
                    w_photo = self.cfg.PHOTOMETRIC_WEIGHT * l_photo
                    w_epi = self.cfg.EPIPOLAR_WEIGHT * l_epi
                    w_smooth = self.cfg.PHY_SMOOTH_WEIGHT * l_smooth
                    w_slope = self.cfg.PHY_SLOPE_WEIGHT * l_slope
                    w_zero = self.cfg.PHY_ZEROMEAN_WEIGHT * l_zeromean
                    loss = w_photo + w_epi + w_smooth + w_slope + w_zero
                    loss = loss / self.cfg.ACCUMULATION_STEPS

                self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                current_loss = loss.item() * self.cfg.ACCUMULATION_STEPS
                ep_stats['total'] += current_loss
                ep_stats['photo'] += l_photo.item()
                ep_stats['epi'] += l_epi.item()
                ep_stats['smooth'] += l_smooth.item()
                ep_stats['slope'] += l_slope.item()
                ep_stats['mean'] += l_zeromean.item()
                count += 1
                pbar.set_postfix({'Loss': current_loss, 'Epi': l_epi.item(), 'Sm': l_smooth.item()})

            if count > 0:
                for k, v in ep_stats.items():
                    if k == 'total':
                        self.history['total_loss'].append(v / count)
                    elif k == 'photo':
                        self.history['photo_loss'].append(v / count)
                    elif k == 'epi':
                        self.history['epi_loss'].append(v / count)
                    elif k == 'smooth':
                        self.history['smooth_loss'].append(v / count)
                    elif k == 'slope':
                        self.history['slope_loss'].append(v / count)
                    elif k == 'mean':
                        self.history['mean_loss'].append(v / count)

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