"""
Complete, runnable DINOv3 + RAFT-style stereo training script.
Based on your 0929 baseline. Additions:
 - Mask-guided correlation gating (suppress background noise)
 - Edge-aware smoothness loss (applied primarily in highlight neighborhoods)
 - Robust local loading of DINOv3 on Windows
 - Full Trainer: training loop, validation, checkpoint, tensorboard logging

Usage:
  - Place this file in your DINOv3 folder and run with your 'dino' conda env.
  - Ensure a local DINOv3 model directory exists at ../dinov3-base-model (relative to DINOv3 folder)
  - Data expected at ../data/left_images and ../data/right_images
"""

import os
import glob
from dataclasses import dataclass
from datetime import datetime
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from transformers import AutoModel


# ---------------- Config ----------------
@dataclass
class Config:
    # The script's directory is considered the project root.
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT: str = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))

    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, 'data', 'left_images')
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, 'data', 'right_images')
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, 'camera_calibration', 'params',
                                         'stereo_calib_params_from_matlab_full.npz')
    # Model is expected to be inside the project root directory.
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, 'dinov3-base-model')

    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 512
    MASK_THRESHOLD: int = 30
    MASK_DILATE: int = 5

    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1

    # MODIFICATION: Reduced iterations to lower sustained GPU load.
    ITERATIONS: int = 4
    DISP_UPDATE_SCALE: float = 1.0

    SMOOTHNESS_WEIGHT: float = 0.05
    PHOTOMETRIC_LOSS_WEIGHTS: tuple = (0.85, 0.15)

    USE_MIXED_PRECISION: bool = True
    GRADIENT_CLIP_VAL: float = 1.0
    # MODIFICATION: Increased gradient accumulation to further improve stability.
    GRADIENT_ACCUMULATION_STEPS: int = 8

    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, 'training_runs_raft')


cfg = Config()


# ---------------- utilities ----------------
def build_highlight_mask(gray_img, thresh=30, dilate=5):
    _, mask = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, k, iterations=1)
    return (mask > 0).astype(np.uint8)


# ---------------- SSIM ----------------
class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.refl = nn.ReflectionPad2d(1)
        self.C1, self.C2 = 0.01 ** 2, 0.03 ** 2

    def forward(self, x, y):
        x, y = self.refl(x), self.refl(y)
        mu_x, mu_y = F.avg_pool2d(x, 3, 1), F.avg_pool2d(y, 3, 1)
        sig_x = F.avg_pool2d(x * x, 3, 1) - mu_x * mu_x
        sig_y = F.avg_pool2d(y * y, 3, 1) - mu_y * mu_y
        sig_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sig_xy + self.C2)
        ssim_d = (mu_x * mu_x + mu_y * mu_y + self.C1) * (sig_x + sig_y + self.C2)
        return torch.clamp((1 - ssim_n / (ssim_d + 1e-8)) / 2, 0, 1)


# ---------------- Dataset ----------------
class RectifiedWaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.is_validation = is_validation
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, '*.*')))
        if not self.left_images:
            raise RuntimeError(f'No left images found in {cfg.LEFT_IMAGE_DIR}')

        # optional calibration
        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            self.roi_left, self.roi_right = tuple(calib['roi_left']), tuple(calib['roi_right'])
            self.has_calib = True
        except Exception:
            self.has_calib = False

        num = len(self.left_images)
        idx = np.arange(num)
        split = int(num * (1 - cfg.VALIDATION_SPLIT))
        if is_validation:
            self.indices = idx[split:]
        else:
            self.indices = idx[:split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        left_path = self.left_images[i]
        base = os.path.basename(left_path)
        # guess right path
        right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, base)
        if not os.path.exists(right_path):
            right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, base.replace('left', 'right').replace('LEFT', 'RIGHT'))

        left_raw = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_raw = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if left_raw is None or right_raw is None:
            return None

        if self.has_calib:
            left_raw = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            right_raw = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
            x, y, w, h = self.roi_left
            left_raw = left_raw[y:y + h, x:x + w]
            x, y, w, h = self.roi_right
            right_raw = right_raw[y:y + h, x:x + w]

        left_rect = cv2.resize(left_raw, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
        right_rect = cv2.resize(right_raw, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))

        mask = build_highlight_mask(left_rect, thresh=self.cfg.MASK_THRESHOLD, dilate=self.cfg.MASK_DILATE)

        def to_tensor(img):
            return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0

        left_t = to_tensor(left_rect)
        right_t = to_tensor(right_rect)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)

        return left_t, right_t, mask_t


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ---------------- Model helper components ----------------
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], 1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], 1)))
        return (1 - z) * h + z * q


class MotionEncoder(nn.Module):
    def __init__(self, corr_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(2 + corr_ch, 128, 3, 2, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)

    def forward(self, disp, corr):
        x = torch.cat([disp, corr], 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.relu(self.conv3(x))


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU(True)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ContextNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, 1)
        self.r1 = ResNetBlock(128, 128)
        self.r2 = ResNetBlock(128, 128)
        # MODIFICATION: Simplified the context network by removing one ResNet block.
        # self.r3 = ResNetBlock(128, 128)
        self.conv_out = nn.Conv2d(128, 256, 1)

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.r1(x)
        x = self.r2(x)
        # x = self.r3(x)
        return self.conv_out(x)


class UpdateBlock(nn.Module):
    def __init__(self, corr_ch, scale=1.0):
        super().__init__()
        self.motion_encoder = MotionEncoder(corr_ch)
        self.gru = ConvGRU(input_dim=128 + 32, hidden_dim=128)
        # MODIFICATION: Simplified the disparity head to reduce computational load.
        self.disp_head = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 1, 1)
        )
        self.scale = scale

    def forward(self, net, inp, corr, disp):
        motion = self.motion_encoder(disp, corr)
        motion_up = F.interpolate(motion, size=inp.shape[-2:], mode='bilinear', align_corners=False)
        inp_cat = torch.cat([inp, motion_up], 1)
        net = self.gru(net, inp_cat)
        delta = torch.tanh(self.disp_head(net)) * self.scale
        return net, delta


# ---------------- CorrBlock with mask gating ----------------
class CorrBlockGated:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        corr = torch.einsum('bchw, bcij->bhwij', fmap1, fmap2)
        corr = corr.flatten(3).permute(0, 3, 1, 2).reshape(-1, 1, fmap2.shape[2], fmap2.shape[3])
        self.corr_pyramid = [corr]
        for _ in range(num_levels - 1):
            self.corr_pyramid.append(F.avg_pool2d(self.corr_pyramid[-1], 2, 2))

    def __call__(self, coords, upsampled_mask=None):
        r = self.radius
        B, _, H, W = coords.shape
        coords = coords.permute(0, 2, 3, 1)
        out = []
        for i, corr in enumerate(self.corr_pyramid):
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), -1)
            centroid = coords.reshape(B * H * W, 1, 1, 2) / (2 ** i)
            coords_lvl = centroid + delta.view(1, -1, 1, 2)
            sampled = F.grid_sample(corr, coords_lvl, padding_mode='border', align_corners=True).view(B, H, W, -1)
            sampled = sampled.permute(0, 3, 1, 2)
            if upsampled_mask is not None:
                m = F.interpolate(upsampled_mask, size=sampled.shape[-2:], mode='bilinear', align_corners=False)
                sampled = sampled * (m + 1e-6)
            out.append(sampled)
        return torch.cat(out, 1).contiguous().float()


# ---------------- DINOv3 Stereo Model (RAFT-style) ----------------
class DINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dino_path = self.cfg.DINO_LOCAL_PATH
        try:
            self.dino = AutoModel.from_pretrained(dino_path, local_files_only=True)
        except Exception as e:
            print(f"[FATAL] loading DINOv3 from {dino_path}: {e}")
            raise
        for p in self.dino.parameters():
            p.requires_grad = False
        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

        # build context and update block sizes
        self.context_net = ContextNetwork(self.feature_dim)
        # compute corr channel size for UpdateBlock's MotionEncoder
        corr_ch = (2 * 4 + 1) ** 2 * 4
        self.update_block = UpdateBlock(corr_ch=corr_ch, scale=self.cfg.DISP_UPDATE_SCALE)

    def get_features(self, image):
        with torch.no_grad():
            outputs = self.dino(image)
            features = outputs.last_hidden_state
        start = 1 + self.num_register_tokens
        patch_tokens = features[:, start:, :]
        b, _, h, w = image.shape
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        return patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, h_feat, w_feat)

    def initialize_flow(self, image):
        b, _, h, w = image.shape
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        return torch.zeros(b, 2, h_feat, w_feat, device=image.device)

    def get_grid(self, b, h, w, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        return torch.stack([x, y], 0).float().unsqueeze(0).repeat(b, 1, 1, 1)

    def forward(self, left_image, right_image, upsampled_mask=None):
        fmap1 = self.get_features(left_image)
        fmap2 = self.get_features(right_image)
        fmap1, fmap2 = F.normalize(fmap1, 2, 1), F.normalize(fmap2, 2, 1)
        corr_fn = CorrBlockGated(fmap1.float(), fmap2.float())
        cnet = self.context_net(fmap1)
        net, inp = torch.split(cnet, [128, 128], 1)
        net, inp = torch.tanh(net), torch.relu(inp)
        flow = self.initialize_flow(left_image)
        b, c, h_feat, w_feat = fmap1.shape
        coords0 = self.get_grid(b, h_feat, w_feat, fmap1.device)
        preds = []
        for _ in range(self.cfg.ITERATIONS):
            flow = flow.detach()
            coords1 = coords0 - flow
            corr = corr_fn(coords1, upsampled_mask)
            net, delta = self.update_block(net, inp, corr, flow)
            flow = flow + delta
            preds.append(flow)
        return preds


# ---------------- Loss: photometric + edge-aware smoothness ----------------
class ImprovedSelfSupervisedLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ssim = SSIM()

    def forward(self, inputs, disp_preds):
        left = inputs['left_image']
        right = inputs['right_image']
        mask = inputs.get('mask', None)
        photo_loss = 0.0
        smooth_loss = 0.0
        for pred in disp_preds:
            disp_up = F.interpolate(pred, size=left.shape[-2:], mode='bilinear', align_corners=False) * (
                    left.shape[2] / pred.shape[2])
            warped = self.inverse_warp(right, disp_up)
            m = mask if mask is not None else torch.ones_like(left[:, 0:1, :, :])
            ssim_l = (self.ssim(warped, left) * m).sum() / (m.sum() + 1e-8)
            l1_l = (torch.abs(warped - left) * m).sum() / (m.sum() + 1e-8)
            photo_loss = photo_loss + (self.cfg.PHOTOMETRIC_LOSS_WEIGHTS[0] * ssim_l + self.cfg.PHOTOMETRIC_LOSS_WEIGHTS[1] * l1_l)
            smooth_loss = smooth_loss + self.edge_aware_smoothness(disp_up, left, mask=m)
        total = photo_loss + self.cfg.SMOOTHNESS_WEIGHT * smooth_loss
        return {'total_loss': total, 'photometric_loss': photo_loss, 'smoothness_loss': smooth_loss, 'warped_right_image': warped}

    def inverse_warp(self, features, disp):
        B, C, H, W = features.shape
        y, x = torch.meshgrid(torch.arange(H, device=features.device), torch.arange(W, device=features.device), indexing='ij')
        grid = torch.stack([x, y], 0).float().unsqueeze(0).repeat(B, 1, 1, 1)
        grid[:, 0] -= disp.squeeze(1)
        grid[:, 0] = 2 * grid[:, 0] / (W - 1) - 1
        grid[:, 1] = 2 * grid[:, 1] / (H - 1) - 1
        return F.grid_sample(features, grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)

    def edge_aware_smoothness(self, disp, img, mask=None):
        # first-order gradients
        disp_dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
        disp_dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]
        img_dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), 1, True)
        img_dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), 1, True)
        w_x = torch.exp(-img_dx)
        w_y = torch.exp(-img_dy)
        if mask is not None:
            mask_x = mask[:, :, :, 1:]
            mask_y = mask[:, :, 1:, :]
            term_x = (disp_dx.abs() * w_x) * mask_x
            term_y = (disp_dy.abs() * w_y) * mask_y
            return term_x.mean() + term_y.mean()
        else:
            return (disp_dx.abs() * w_x).mean() + (disp_dy.abs() * w_y).mean()


# ---------------- Evaluation ----------------
class EvaluationMetrics:
    @staticmethod
    def compute_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    @staticmethod
    def evaluate_reconstruction(inputs, outputs, loss_components):
        left_img = inputs['left_image']
        warped = loss_components.get('warped_right_image', None)
        if warped is None:
            return {'psnr': 0.0}
        # Detach the 'warped' tensor before metric calculation
        # to prevent any autograd-related side effects.
        return {'psnr': EvaluationMetrics.compute_psnr(left_img, warped.detach())}


# ---------------- Trainer ----------------
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'visualizations'), exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('✓ Using device:', self.device)

        train_ds = RectifiedWaveStereoDataset(cfg, is_validation=False)
        val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)
        # MODIFICATION: Set num_workers to 0 to reduce system overhead and improve stability,
        # which can help prevent hardware-related reboots on Windows.
        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                                       num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=0, pin_memory=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, 'tensorboard')) if SummaryWriter else None
        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.loss_fn = ImprovedSelfSupervisedLoss(cfg)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE,
                                     weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=cfg.LEARNING_RATE,
                                                       epochs=cfg.NUM_EPOCHS, steps_per_epoch=max(1, len(self.train_loader) // cfg.GRADIENT_ACCUMULATION_STEPS))
        self.evaluator = EvaluationMetrics()
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION)

        print('LEFT_IMAGE_DIR =', cfg.LEFT_IMAGE_DIR)
        print('RIGHT_IMAGE_DIR =', cfg.RIGHT_IMAGE_DIR)
        print('DINO_LOCAL_PATH =', cfg.DINO_LOCAL_PATH)

        self.step = 0
        self.best_val = -1e9
        self.log_file = os.path.join(self.run_dir, 'logs', 'history.json')

    def _pad(self, left, right, mask):
        _, _, h, w = left.shape
        pad_h = (self.model.patch_size - h % self.model.patch_size) % self.model.patch_size
        pad_w = (self.model.patch_size - w % self.model.patch_size) % self.model.patch_size
        if pad_h > 0 or pad_w > 0:
            left = F.pad(left, (0, pad_w, 0, pad_h), mode='constant', value=0)
            right = F.pad(right, (0, pad_w, 0, pad_h), mode='constant', value=0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return left, right, mask

    def train(self):
        history = {'train': [], 'val': []}
        for epoch in range(self.cfg.NUM_EPOCHS):
            train_stats = self._run_epoch(epoch, True)
            val_stats = self._run_epoch(epoch, False)
            history['train'].append(train_stats)
            history['val'].append(val_stats)
            with open(self.log_file, 'w') as f:
                json.dump(history, f, indent=2)
            if val_stats['psnr'] > self.best_val:
                self.best_val = val_stats['psnr']
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, 'checkpoints', 'best.pth'))
                print('✓ Saved best model')
        if self.writer:
            self.writer.close()
        print('Training finished')

    def _run_epoch(self, epoch, training=True):
        if training:
            self.model.train()
            loader = self.train_loader
            desc = f'Epoch {epoch+1}/{self.cfg.NUM_EPOCHS} [Train]'
        else:
            self.model.eval()
            loader = self.val_loader
            desc = f'Epoch {epoch+1}/{self.cfg.NUM_EPOCHS} [Val]'

        pbar = tqdm(loader, desc=desc)
        total_loss = 0.0
        total_ph = 0.0
        total_sm = 0.0
        total_psnr = 0.0
        n = 0

        if training:
            self.optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            if batch is None:
                continue
            left, right, mask = [x.to(self.device) for x in batch]
            left, right, mask = self._pad(left, right, mask)

            up_mask = F.interpolate(mask, size=(left.shape[2] // self.model.patch_size, left.shape[3] // self.model.patch_size), mode='bilinear', align_corners=False)

            with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION):
                preds = self.model(left, right, up_mask)
                disp_preds = [p[:, 0:1, :, :] for p in preds]
                loss_comps = self.loss_fn({'left_image': left, 'right_image': right, 'mask': mask}, disp_preds)
                loss = loss_comps['total_loss']

            if training:
                if torch.isfinite(loss):
                    scaled_loss = loss / self.cfg.GRADIENT_ACCUMULATION_STEPS
                    self.scaler.scale(scaled_loss).backward()

                    if (i + 1) % self.cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), cfg.GRADIENT_CLIP_VAL)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    print('Warning: non-finite loss, skipping update')

            if training and (i + 1) % self.cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            metrics = self.evaluator.evaluate_reconstruction({'left_image': left}, None, loss_comps)
            psnr = metrics.get('psnr', 0.0)
            total_loss += loss.item()
            total_ph += loss_comps.get('photometric_loss', torch.tensor(0.0)).item()
            total_sm += loss_comps.get('smoothness_loss', torch.tensor(0.0)).item()
            total_psnr += psnr
            n += 1
            pbar.set_postfix({'loss': loss.item(), 'psnr': psnr})
            if not training:
                self.step +=1 # just for validation tracking

        if training:
            self.step = (epoch + 1) * len(loader)

        avg = {'total': total_loss / max(1, n), 'photometric': total_ph / max(1, n), 'smoothness': total_sm / max(1, n), 'psnr': total_psnr / max(1, n)}
        print(f"{desc} -> loss: {avg['total']:.4f}, psnr: {avg['psnr']:.2f}")
        return avg


# ---------------- main ----------------
if __name__ == '__main__':
    # Add CUDA launch blocking for better error reporting if needed for debugging
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print('Starting training with edge-aware smoothness + mask-guided correlation...')
    trainer = Trainer(cfg)
    trainer.train()

