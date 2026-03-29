# self_supervised_dinov3_raft.py
# An end-to-end, self-supervised 3D reconstruction pipeline for dynamic wave surfaces.
# FINAL ARCHITECTURE: RAFT-style iterative refinement with ConvGRU update.
# FINAL OPTIMIZATIONS:
# 1. Added Dropout to the ContextNetwork for regularization to combat overfitting.
# 2. Refined the MotionEncoder architecture for better stability and feature processing.
# 3. Adjusted the auto-tuner to a more conservative profile to ensure hardware stability
#    and prevent system restarts.

import os
import sys
import glob
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import argparse
import subprocess
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import exp

# --- TensorBoard Import ---
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# --- Hugging Face Transformers Import ---
try:
    from transformers import AutoModel
except ImportError:
    print(
        "=" * 80 + "\n[FATAL ERROR]: 'transformers' library not found. Please run: pip install transformers accelerate\n" + "=" * 80);
    sys.exit(1)

# --- 1. Configuration Center ---
PROJECT_ROOT = r"D:\Research\wave_reconstruction_project\DINOv3"
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


@dataclass
class Config:
    """Project Configuration Parameters"""
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                         "stereo_calib_params_from_matlab_full.npz")
    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs_raft")
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")

    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 400
    IMAGE_HEIGHT: int = 256
    IMAGE_WIDTH: int = 512
    MASK_THRESHOLD: int = 30

    BATCH_SIZE: int = 1
    # OPTIMIZATION: Further lowered learning rate for more stable convergence.
    LEARNING_RATE: float = 1e-5
    NUM_EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    GRADIENT_ACCUMULATION_STEPS: int = 1

    USE_MIXED_PRECISION: bool = True
    PHOTOMETRIC_LOSS_WEIGHTS: tuple = (0.85, 0.15)
    # OPTIMIZATION: Drastically reduced smoothness weight to prevent model from collapsing to zero disparity.
    SMOOTHNESS_WEIGHT: float = 0.05

    USE_ADVANCED_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.8

    # OPTIMIZATION: Increased patience to give the model more time to learn with the new parameters.
    EARLY_STOPPING_PATIENCE: int = 20

    ITERATIONS: int = 8
    DEEP_SUPERVISION_DECAY: float = 0.8
    # OPTIMIZATION: Reduced disparity update scale to prevent extreme warping.
    DISP_UPDATE_SCALE: float = 1.0


# --- Auto hardware tuning helpers ---
def auto_tune_config(cfg: Config):
    print("--- Probing GPU for auto-tuning ---")
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      stderr=subprocess.DEVNULL).decode("utf-8").strip()
        free_mem_mb = int(out)
    except Exception:
        free_mem_mb = None

    patch_size = 16
    if not free_mem_mb:
        print("[WARNING] Could not probe GPU. Using conservative default settings.")
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT = 1, 4, 384, 192
        return cfg

    print(f"✓ Available GPU Memory: {free_mem_mb} MB")

    # [HARDWARE-SAFE] More conservative tuning
    if free_mem_mb < 8000:
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 4, 4, 0.7
    elif free_mem_mb < 12000:
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 1, 6, 2, 0.8
    elif free_mem_mb < 20000:
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 2, 6, 1, 0.9
    else:
        cfg.BATCH_SIZE, cfg.ITERATIONS, cfg.GRADIENT_ACCUMULATION_STEPS, scale = 2, 8, 1, 1.0

    cfg.IMAGE_WIDTH = int((cfg.IMAGE_WIDTH * scale) // patch_size) * patch_size
    cfg.IMAGE_HEIGHT = int((cfg.IMAGE_HEIGHT * scale) // patch_size) * patch_size

    print(
        "--- Auto-tuned config applied: ---\n" + f"  Resolution: {cfg.IMAGE_WIDTH}x{cfg.IMAGE_HEIGHT}, Batch Size: {cfg.BATCH_SIZE}, Iterations: {cfg.ITERATIONS}, Grad Accum: {cfg.GRADIENT_ACCUMULATION_STEPS}\n" + "---------------------------------")
    return cfg


# --- 2. Core Model Components ---
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], 1)
        z, r = torch.sigmoid(self.convz(hx)), torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], 1)))
        return (1 - z) * h + z * q


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        corr_ch = (2 * 4 + 1) ** 2 * 4
        # FIX: The input is a 2-channel flow tensor plus the correlation volume.
        # Changed '1 + corr_ch' to '2 + corr_ch' to match the actual input dimensions.
        self.conv1 = nn.Sequential(nn.Conv2d(2 + corr_ch, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.ReLU(True))

    def forward(self, disp, corr):
        x = torch.cat([disp, corr], 1)
        return self.conv3(self.conv2(self.conv1(x)))


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout_prob=0.1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout2d(dropout_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.GroupNorm(8, out_ch))

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ContextNetwork(nn.Module):
    def __init__(self, in_channels):
        super(ContextNetwork, self).__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(in_channels, 256, 1), nn.ReLU(True))
        self.layer1 = ResNetBlock(256, 128)
        self.layer2 = ResNetBlock(128, 128)
        self.conv_out = nn.Conv2d(128, 256, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.conv_out(x)


class UpdateBlock(nn.Module):
    def __init__(self, scale=10.0):
        super(UpdateBlock, self).__init__()
        self.motion_encoder = MotionEncoder()
        self.gru = ConvGRU(input_dim=128 + 32, hidden_dim=128)
        self.disp_head = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True), nn.Conv2d(256, 1, 1))
        self.scale = scale

    def forward(self, net, inp, corr, disp):
        motion_features = self.motion_encoder(disp, corr)
        inp_cat = torch.cat(
            [inp, F.interpolate(motion_features, size=inp.shape[-2:], mode='bilinear', align_corners=False)], 1)
        net = self.gru(net, inp_cat)
        # The scale parameter is now controlled by cfg.DISP_UPDATE_SCALE
        return net, torch.tanh(self.disp_head(net)) * self.scale


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels, self.radius = num_levels, radius
        self.corr_pyramid = []
        corr = torch.einsum('bchw, bcij->bhwij', fmap1, fmap2)
        corr = corr.flatten(3).permute(0, 3, 1, 2).reshape(-1, 1, fmap2.shape[2], fmap2.shape[3])
        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1): self.corr_pyramid.append(F.avg_pool2d(self.corr_pyramid[-1], 2, 2))

    def __call__(self, coords):
        r, B, _, H, W = self.radius, *coords.shape
        coords = coords.permute(0, 2, 3, 1)
        out = []
        for i, corr in enumerate(self.corr_pyramid):
            dx, dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device), torch.linspace(-r, r, 2 * r + 1,
                                                                                            device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), -1)
            centroid = coords.reshape(B * H * W, 1, 1, 2) / 2 ** i
            coords_lvl = centroid + delta.view(1, -1, 1, 2)
            out.append(F.grid_sample(corr, coords_lvl, padding_mode='border', align_corners=True).view(B, H, W, -1))
        return torch.cat(out, -1).permute(0, 3, 1, 2).contiguous().float()


# --- 3. Self-Supervised Loss Function ---
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.refl = nn.ReflectionPad2d(1)
        self.C1, self.C2 = 0.01 ** 2, 0.03 ** 2

    def forward(self, x, y):
        x, y = self.refl(x), self.refl(y)
        mu_x, mu_y = F.avg_pool2d(x, 3, 1), F.avg_pool2d(y, 3, 1)
        sig_x, sig_y = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2, F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sig_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sig_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sig_x + sig_y + self.C2)
        return torch.clamp((1 - SSIM_n / (SSIM_d + 1e-8)) / 2, 0, 1)


class ImprovedSelfSupervisedLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.photometric_weights = cfg.PHOTOMETRIC_LOSS_WEIGHTS
        self.ssim = SSIM()

    def forward(self, inputs, disp_preds):
        left, right, mask = inputs["left_image"], inputs["right_image"], inputs["mask"]
        photo_loss, smooth_loss = 0.0, 0.0
        for i, pred in enumerate(disp_preds):
            i_w = self.cfg.DEEP_SUPERVISION_DECAY ** (self.cfg.ITERATIONS - i - 1)
            disp = F.interpolate(pred, size=left.shape[-2:], mode='bilinear', align_corners=False) * (
                        left.shape[2] / pred.shape[2])
            warped = self.inverse_warp(right, disp)
            photo_loss += i_w * self.compute_photometric_loss(warped, left, mask)
            smooth_loss += i_w * self.compute_smoothness_loss(disp, left)
        total_loss = photo_loss + self.cfg.SMOOTHNESS_WEIGHT * smooth_loss
        final_disp_up = F.interpolate(disp_preds[-1], size=left.shape[-2:], mode='bilinear', align_corners=False) * (
                    left.shape[2] / disp_preds[-1].shape[2])
        return {"total_loss": total_loss, "photometric_loss": photo_loss, "smoothness_loss": smooth_loss,
                "warped_right_image": self.inverse_warp(right, final_disp_up)}

    def compute_photometric_loss(self, w, t, m): return self.photometric_weights[0] * (self.ssim(w, t) * m).sum() / (
                m.sum() + 1e-8) + self.photometric_weights[1] * (torch.abs(w - t) * m).sum() / (m.sum() + 1e-8)

    def inverse_warp(self, features, disp):
        B, C, H, W = features.shape
        y, x = torch.meshgrid(torch.arange(H, device=features.device), torch.arange(W, device=features.device),
                              indexing='ij')
        grid = torch.stack([x, y], 0).float().repeat(B, 1, 1, 1)
        grid[:, 0] -= disp.squeeze(1)
        grid[:, 0] = 2 * grid[:, 0] / (W - 1) - 1;
        grid[:, 1] = 2 * grid[:, 1] / (H - 1) - 1
        return F.grid_sample(features, grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border',
                             align_corners=True)

    def compute_smoothness_loss(self, disp, img):
        disp_dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
        disp_dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]
        img_dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), 1, True)
        img_dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), 1, True)
        disp_dxx = disp_dx[:, :, :, 1:] - disp_dx[:, :, :, :-1]
        disp_dyy = disp_dy[:, :, 1:, :] - disp_dy[:, :, :-1, :]
        return (disp_dxx.abs() * torch.exp(-img_dx[:, :, :, :-1])).mean() + (
                    disp_dyy.abs() * torch.exp(-img_dy[:, :, :-1, :])).mean()


# --- 4. PyTorch Dataset ---
class RectifiedWaveStereoDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg, self.is_validation = cfg, is_validation
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images: sys.exit(f"No images found in '{cfg.LEFT_IMAGE_DIR}'.")
        try:
            calib = np.load(cfg.CALIBRATION_FILE)
            self.map1_left, self.map2_left = calib['map1_left'], calib['map2_left']
            self.map1_right, self.map2_right = calib['map1_right'], calib['map2_right']
            self.roi_left, self.roi_right = tuple(calib['roi_left']), tuple(calib['roi_right'])
        except Exception as e:
            sys.exit(f"Failed to load calibration file: {e}")
        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        try:
            left_path = self.left_images[self.indices[idx]]
            right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, 'right' + os.path.basename(left_path)[4:])
            left_raw, right_raw = cv2.imread(left_path, 0), cv2.imread(right_path, 0)
            if left_raw is None or right_raw is None: return None
            left_rect = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
            x, y, w, h = self.roi_left;
            left_rect = left_rect[y:y + h, x:x + w]
            x, y, w, h = self.roi_right;
            right_rect = right_rect[y:y + h, x:x + w]
            left_img = cv2.resize(left_rect, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
            right_img = cv2.resize(right_rect, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
            if not self.is_validation and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
                if np.random.rand() < 0.5:
                    brightness = np.random.uniform(0.7, 1.3)
                    left_img, right_img = [np.clip(im * brightness, 0, 255).astype(np.uint8) for im in
                                           [left_img, right_img]]
                if np.random.rand() < 0.5:
                    contrast = np.random.uniform(0.7, 1.3)
                    left_img = np.clip((left_img - left_img.mean()) * contrast + left_img.mean(), 0, 255).astype(
                        np.uint8)
                    right_img = np.clip((right_img - right_img.mean()) * contrast + right_img.mean(), 0, 255).astype(
                        np.uint8)
                if np.random.rand() < 0.5:
                    left_img, right_img = cv2.flip(left_img, 1), cv2.flip(right_img, 1)
            _, mask = cv2.threshold(left_img, self.cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

            def to_tensor(img):
                return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float() / 255.0

            return to_tensor(left_img), to_tensor(right_img), torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        except Exception as e:
            print(f"Warning at idx {idx}: {e}"); return None


# --- 5. RAFT-style DINOv3 Stereo Model ---
class DINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.dino = self._load_dino_model()
        for p in self.dino.parameters(): p.requires_grad = False
        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)
        self.context_net = ContextNetwork(self.feature_dim)
        self.update_block = UpdateBlock(scale=self.cfg.DISP_UPDATE_SCALE)
        print("✓ RAFT-style model with Context Network built (DINOv3 FROZEN).")

    def _load_dino_model(self):
        try:
            return AutoModel.from_pretrained(self.cfg.DINO_LOCAL_PATH, local_files_only=True)
        except Exception as e:
            print(f"[FATAL] loading DINOv3: {e}"); sys.exit(1)

    def get_features(self, image):
        with torch.no_grad(): features = self.dino(image).last_hidden_state
        b, _, h, w = image.shape
        start_idx = 1 + self.num_register_tokens
        patch_tokens = features[:, start_idx:, :]
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        return patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, h_feat, w_feat)

    def initialize_flow(self, image):
        b, _, h, w = image.shape
        h_feat, w_feat = h // self.patch_size, w // self.patch_size
        return torch.zeros(b, 2, h_feat, w_feat, device=image.device)

    def get_grid(self, b, h, w, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        return torch.stack([x, y], 0).float().unsqueeze(0).repeat(b, 1, 1, 1)

    def forward(self, left_image, right_image):
        fmap1, fmap2 = self.get_features(left_image), self.get_features(right_image)
        fmap1, fmap2 = F.normalize(fmap1, 2, 1), F.normalize(fmap2, 2, 1)
        corr_fn = CorrBlock(fmap1.float(), fmap2.float())
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
            corr = corr_fn(coords1)
            net, delta_flow = self.update_block(net, inp, corr, flow)
            flow = flow + delta_flow
            preds.append(flow)
        return preds


# --- Evaluation and Trainer ---
class EvaluationMetrics:
    @staticmethod
    def compute_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2);
        return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    @staticmethod
    def evaluate_reconstruction(inputs, outputs, loss_components):
        left_img = inputs["left_image"]
        if "warped_right_image" not in loss_components or loss_components["warped_right_image"] is None:
            return {"psnr": 0.0}
        return {"psnr": EvaluationMetrics.compute_psnr(left_img, loss_components["warped_right_image"])}


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR, self.timestamp)
        for d in ["checkpoints", "visualizations", "logs", "tensorboard"]: os.makedirs(os.path.join(self.run_dir, d),
                                                                                       exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        print(f"✓ Using device: {self.device}")

        train_ds, val_ds = RectifiedWaveStereoDataset(cfg, False), RectifiedWaveStereoDataset(cfg, True)
        self.train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                                       num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=0, pin_memory=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tensorboard")) if SummaryWriter else None
        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.loss_fn = ImprovedSelfSupervisedLoss(cfg)

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE,
                                     weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.cfg.LEARNING_RATE,
                                                       epochs=self.cfg.NUM_EPOCHS,
                                                       steps_per_epoch=len(self.train_loader))

        self.evaluator = EvaluationMetrics()
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION)
        self.step, self.log_file = 0, os.path.join(self.run_dir, "logs", "training_log.json")
        self.loss_keys, self.metric_keys = ['total', 'photometric', 'smoothness'], ['psnr']
        self.history = {'train': {k: [] for k in self.loss_keys + self.metric_keys},
                        'val': {k: [] for k in self.loss_keys + self.metric_keys}}

    def train(self):
        print("\n--- Starting RAFT-style Self-Supervised Training (DINOv3 Frozen) ---")
        best_val_psnr, epochs_no_improve = 0.0, 0
        for epoch in range(self.cfg.NUM_EPOCHS):
            train_results = self._run_epoch(epoch, True)
            self._log_epoch_results('train', epoch, train_results)
            with torch.no_grad():
                val_results = self._run_epoch(epoch, False)
            self._log_epoch_results('val', epoch, val_results)
            avg_val_psnr = val_results.get('psnr', 0.0)
            print(
                f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> Train Loss: {train_results.get('total', 0.0):.4f} | Val PSNR: {avg_val_psnr:.2f}")
            self.update_log_file(epoch)
            if self.cfg.VISUALIZE_TRAINING: self.plot_training_history()
            if avg_val_psnr > best_val_psnr:
                best_val_psnr, epochs_no_improve = avg_val_psnr, 0
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, "checkpoints", "best_model_raft.pth"))
                print(f"✓ Val PSNR improved. Model saved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.EARLY_STOPPING_PATIENCE: print(
                    f"--- Early stopping after {epoch + 1} epochs. ---"); break
        print("--- Training complete! ---")
        if self.writer: self.writer.close()

    def _pad_inputs(self, left, right, mask):
        _, _, h, w = left.shape
        pad_h = (self.model.patch_size - h % self.model.patch_size) % self.model.patch_size
        pad_w = (self.model.patch_size - w % self.model.patch_size) % self.model.patch_size
        if pad_h > 0 or pad_w > 0:
            return [F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0) for x in [left, right, mask]]
        return left, right, mask

    def _run_epoch(self, epoch, is_training):
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [{'Train' if is_training else 'Val'}]")
        epoch_results = {k: 0.0 for k in self.loss_keys + self.metric_keys}
        for data in pbar:
            if data is None: continue
            left, right, mask = [d.to(self.device) for d in data]
            left, right, mask = self._pad_inputs(left, right, mask)
            with torch.amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION):
                flow_preds = self.model(left, right)
                disp_preds = [flow[:, 0:1, :, :] for flow in flow_preds]
                loss_comps = self.loss_fn({"left_image": left, "right_image": right, "mask": mask}, disp_preds)
                loss = loss_comps["total_loss"]
            if is_training:
                if torch.isfinite(loss):
                    accum_steps = self.cfg.GRADIENT_ACCUMULATION_STEPS
                    self.scaler.scale(loss / accum_steps).backward()
                    if (self.step + 1) % accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    # BUG FIX: Moved scheduler step to after optimizer step.
                    self.scheduler.step()
                else:
                    print(f"\nWarning: Invalid loss at step {self.step}. Skipping.")

            final_disp = disp_preds[-1]
            metrics = self.evaluator.evaluate_reconstruction({"left_image": left}, {"disparity": final_disp},
                                                             loss_comps)
            for k in self.loss_keys:
                if f"{k}_loss" in loss_comps: epoch_results[k] += loss_comps[f"{k}_loss"].item()
            for k in self.metric_keys:
                if k in metrics: epoch_results[k] += metrics[k]
            pbar.set_postfix({'loss': loss.item(), 'psnr': metrics.get('psnr', 0.0)})
            if is_training:
                if self.writer: self.writer.add_scalar('Loss/step_train', loss.item(), self.step)
                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0: self.visualize(left,
                                                                                                                mask,
                                                                                                                final_disp,
                                                                                                                loss_comps,
                                                                                                                self.step,
                                                                                                                "train")
                self.step += 1

        num_batches = len(loader)
        return {k: v / num_batches for k, v in epoch_results.items()} if num_batches > 0 else epoch_results

    def _log_epoch_results(self, phase, epoch, results):
        for k, v in results.items():
            self.history[phase][k].append(v)
            if self.writer: self.writer.add_scalar(f"{'Loss' if k in self.loss_keys else 'Metrics'}/{phase}_{k}", v,
                                                   epoch)
        if self.writer and phase == 'val' and self.history['train']['total']:
            self.writer.add_scalars('Loss/epoch_comparison',
                                    {'train': self.history['train']['total'][-1], 'val': results.get('total', 0)},
                                    epoch)
            if 'psnr' in self.history['train'] and self.history['train']['psnr']: self.writer.add_scalars(
                'PSNR/epoch_comparison', {'train': self.history['train']['psnr'][-1], 'val': results.get('psnr', 0)},
                epoch)

    def visualize(self, left_tensor, mask_tensor, final_disp, loss_components, step, phase):
        left_img = (left_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = mask_tensor[0, 0].cpu().numpy()
        pred_disp_up = F.interpolate(final_disp, size=left_img.shape[:2], mode='bilinear', align_corners=False)
        pred_disp = pred_disp_up[0, 0].cpu().detach().numpy()
        masked_disp = np.ma.masked_where(mask == 0, pred_disp)
        warped_right = (loss_components["warped_right_image"][0].cpu().permute(1, 2, 0).detach().numpy() * 255).astype(
            np.uint8)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'RAFT Model - Step: {step} ({phase})', fontsize=16)
        axes[0, 0].imshow(left_img);
        axes[0, 0].set_title("Input Left Image");
        axes[0, 0].axis('off')
        axes[0, 1].imshow(warped_right);
        axes[0, 1].set_title("Reconstructed Left from Right");
        axes[0, 1].axis('off')
        im = axes[1, 0].imshow(masked_disp, cmap='viridis');
        axes[1, 0].set_title("Predicted Disparity (Masked)")
        axes[1, 0].set_facecolor('black');
        axes[1, 0].axis('off');
        fig.colorbar(im, ax=axes[1, 0])
        diff_map = np.abs(left_img.astype(float) - warped_right.astype(float)).mean(2)
        im_diff = axes[1, 1].imshow(diff_map, cmap='hot');
        axes[1, 1].set_title("Photometric Error")
        axes[1, 1].axis('off');
        fig.colorbar(im_diff, ax=axes[1, 1])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.run_dir, "visualizations", f"{phase}_step_{step:06d}.png")
        plt.savefig(save_path)
        if self.writer: self.writer.add_figure(f'Visualization/{phase}', fig, step)
        plt.close(fig)

    def plot_training_history(self):
        if not self.history['train']['total']: return
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('RAFT Model Training History', fontsize=16)
        axes[0].plot(self.history['train']['total'], label='Train Loss');
        axes[0].plot(self.history['val']['total'], label='Val Loss')
        axes[0].set_title('Total Loss');
        axes[0].legend();
        axes[0].grid(True);
        axes[0].set_xlabel("Epochs");
        axes[0].set_ylabel("Loss")
        axes[1].plot(self.history['train']['psnr'], label='Train PSNR');
        axes[1].plot(self.history['val']['psnr'], label='Val PSNR')
        axes[1].set_title('PSNR');
        axes[1].legend();
        axes[1].grid(True);
        axes[1].set_xlabel("Epochs");
        axes[1].set_ylabel("PSNR (dB)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.run_dir, "visualizations", "training_history.png")
        plt.savefig(save_path)
        plt.close(fig)

    def update_log_file(self, epoch):
        log_data = {'config': asdict(self.cfg), 'epoch': epoch, 'history': self.history}
        with open(self.log_file, 'w') as f: json.dump(log_data, f, indent=2, default=str)


if __name__ == "__main__":
    cfg = Config()
    try:
        auto_tune_config(cfg)
    except Exception as e:
        print(f"[WARNING] Auto-tuning failed: {e}. Using default config.")

    if cfg.BATCH_SIZE == 0:
        print("[WARNING] Auto-tuned batch size is 0. Setting to 1.")
        cfg.BATCH_SIZE = 1

    trainer = Trainer(cfg)
    trainer.train()

