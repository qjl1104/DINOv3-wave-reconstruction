# -*- coding: utf-8 -*-
"""
DINOv3 Stereo Wave Reconstruction (ViT-B/16)
- 原始分辨率 + 分块提特征（tile）+ 汉宁窗融合
- 自动对齐 tile/overlap 至 patch 的倍数；同一配置只打印一次提示
- 成对一致的数据增强（亮度/对比度/噪声）
- AMP 新 API（torch.amp）
- 通道归一化后再建体积（更稳）
- 成本体积支持两种度量：差分("diff") / 相关("corr")，默认 corr
- 光度损失加入越界/遮挡掩膜 + 梯度域 L1
- cap 命中率监控（视差是否“顶天花板”）
- 分目录保存（避免与旧实验串味）
- 训练/验证可视化，历史曲线保存；验证阶段控制台打印损失分解
"""

import os, sys, glob, json
from dataclasses import dataclass, asdict
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from math import exp

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from transformers import AutoModel
except Exception:
    print("无法导入 transformers.AutoModel，请先安装：pip install transformers")
    raise

from torch import amp as _amp


# =========================
# 1. 配置中心（按需修改数据路径）
# =========================
@dataclass
class Config:
    # 数据路径（你的左右图像已完成标定 + 极线校正）
    LEFT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/lresult/"
    RIGHT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/rresult/"

    # DINOv3 ViT-B/16 预训练权重（你使用的）
    DINO_ONLINE_MODEL: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    DINO_LOCAL_PATH: str = "./dinov3-vitb16-lvd1689m/"   # 新目录，避免读到旧缓存

    # 分块/分辨率策略
    PRESERVE_ORIGINAL_RES: bool = True  # True: 使用原图尺寸并走分块提特征
    USE_TILING: bool = True
    TILE_HEIGHT: int = 960     # 60×16
    TILE_WIDTH: int  = 1024    # 64×16
    TILE_OVERLAP: int = 128    # 8×16
    PAD_TO_PATCH: bool = True  # 右/下补零到 patch 对齐

    # 仅当不保留原始分辨率时使用（快速试跑）
    IMAGE_HEIGHT: int = 252
    IMAGE_WIDTH: int  = 504

    # 训练相关
    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1
    USE_MIXED_PRECISION: bool = True
    GRADIENT_CLIP_VAL: float = 1.0

    # 视差范围（像素单位；会内部转换为特征层大小）
    MAX_DISPARITY: int = 160

    # 成本体积类型："diff"（差分）或 "corr"（相关）。归一化特征建议用 "corr"
    COST_TYPE: str = "corr"

    # 自监督损失
    PHOTOMETRIC_LOSS_WEIGHTS: tuple = (0.7, 0.3)  # (SSIM, L1)
    INITIAL_SMOOTHNESS_WEIGHT: float = 0.1
    SMOOTHNESS_WEIGHT_DECAY: float = 0.95
    USE_CONSISTENCY_LOSS: bool = True
    CONSISTENCY_LOSS_WEIGHT: float = 0.1

    # 增强（仅训练集；成对一致）
    USE_DATA_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.3

    # 可视化
    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 100

    # 运行标签（空则自动生成），用于分目录保存
    RUN_TAG: str = ""

    # 目录（会在 main 里根据 RUN_TAG 被重写到 ./experiments/<RUN_TAG>/...）
    CHECKPOINT_DIR: str = "./experiments/default/checkpoints/"
    VISUALIZATION_DIR: str = "./experiments/default/visualization/"
    LOG_DIR: str = "./experiments/default/logs/"
    TENSORBOARD_DIR: str = "./experiments/default/runs/"


# =========================
# 2. 自监督损失
# =========================
class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', self.gaussian(11, 1.5))
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def forward(self, x, y):
        x = self.refl(x); y = self.refl(y)
        mu_x = self.mu_x_pool(x); mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        # 输出为“DSSIM”图（0好 1差）
        return torch.clamp((1 - SSIM_n / (SSIM_d + 1e-8)) / 2, 0, 1)


class ImprovedSelfSupervisedLoss(nn.Module):
    def __init__(self, smoothness_weight=0.1, photometric_weights=(0.7, 0.3),
                 use_consistency_loss=True, consistency_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.photometric_weights = photometric_weights
        self.use_consistency_loss = use_consistency_loss
        self.consistency_weight = consistency_weight
        self.ssim = SSIM()

    def forward(self, inputs, outputs):
        left_img = inputs["left_image"]
        right_img = inputs["right_image"]
        pred_disp = outputs["disparity"]

        warped_right_image = self.inverse_warp(right_img, pred_disp)   # 右→左
        warped_left_image  = self.inverse_warp(left_img, -pred_disp)   # 左→右

        # --- 有效掩膜（仅需 x 方向，极线校正后 y 不变） ---
        B, _, H, W = left_img.shape
        device = left_img.device
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'
        )
        disp = pred_disp.squeeze(1)
        tx_L2R = x_coords - disp          # 左像素在右图的采样位置
        tx_R2L = x_coords + disp          # 右像素在左图的采样位置
        valid_L2R = ((tx_L2R >= 0) & (tx_L2R <= W - 1)).unsqueeze(0).unsqueeze(0).float()
        valid_R2L = ((tx_R2L >= 0) & (tx_R2L <= W - 1)).unsqueeze(0).unsqueeze(0).float()

        # --- 光度损失：加权 L1 + 加权 SSIM ---
        eps = 1e-6
        l1_r = (torch.abs(warped_right_image - left_img) * valid_L2R).sum() / (valid_L2R.sum() + eps)
        l1_l = (torch.abs(warped_left_image  - right_img) * valid_R2L).sum() / (valid_R2L.sum() + eps)
        l1_loss = 0.5 * (l1_r + l1_l)

        ssim_r = (self.ssim(warped_right_image, left_img) * valid_L2R).sum() / (valid_L2R.sum() + eps)
        ssim_l = (self.ssim(warped_left_image,  right_img) * valid_R2L).sum() / (valid_R2L.sum() + eps)
        ssim_loss = 0.5 * (ssim_r + ssim_l)

        # --- 额外：梯度域光度项（Sobel），更鲁棒于反射/亮度漂移 ---
        def sobel_xy(img):
            Gx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
            Gy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
            pad = (1,1,1,1)
            gx = F.conv2d(F.pad(img, pad, mode='replicate'), Gx.repeat(img.size(1),1,1,1), groups=img.size(1))
            gy = F.conv2d(F.pad(img, pad, mode='replicate'), Gy.repeat(img.size(1),1,1,1), groups=img.size(1))
            g = torch.sqrt(gx*gx + gy*gy + 1e-6).mean(1, keepdim=True)  # 通道平均
            return g

        grad_l = sobel_xy(left_img);  grad_r = sobel_xy(right_img)
        grad_warp_r = sobel_xy(warped_right_image)
        grad_warp_l = sobel_xy(warped_left_image)

        gl1_r = (torch.abs(grad_warp_r - grad_l) * valid_L2R).sum() / (valid_L2R.sum() + eps)
        gl1_l = (torch.abs(grad_warp_l - grad_r) * valid_R2L).sum() / (valid_R2L.sum() + eps)
        grad_l1_loss = 0.5 * (gl1_r + gl1_l)

        photometric_loss = (
            self.photometric_weights[0] * ssim_loss +
            self.photometric_weights[1] * l1_loss +
            0.10 * grad_l1_loss
        )

        # --- 平滑损失 ---
        smoothness_loss = self.compute_smoothness_loss(pred_disp, left_img)

        # --- 一致性损失（可选） ---
        consistency_loss = 0
        if self.use_consistency_loss and outputs.get("disparity_right", None) is not None:
            disp_left  = outputs["disparity"]
            disp_right = outputs["disparity_right"]
            warped_disp_right = self.inverse_warp(disp_right, -disp_left)
            consistency_loss = torch.abs(disp_left - warped_disp_right).mean()

        total_loss = photometric_loss + self.smoothness_weight * smoothness_loss
        if self.use_consistency_loss:
            total_loss += self.consistency_weight * consistency_loss

        return {
            "total_loss": total_loss,
            "photometric_loss": photometric_loss,
            "smoothness_loss": smoothness_loss,
            "consistency_loss": consistency_loss,
            "warped_right_image": warped_right_image,
            "warped_left_image": warped_left_image
        }

    def inverse_warp(self, features, disp):
        B, C, H, W = features.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=features.device),
            torch.arange(W, device=features.device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords, y_coords], dim=0).float().repeat(B, 1, 1, 1)
        disp = disp.squeeze(1)
        transformed_x = pixel_coords[:, 0, :, :] - disp
        grid = torch.stack([transformed_x, pixel_coords[:, 1, :, :]], dim=-1)
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        # zeros padding，避免“拉边框”污染光度项
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    def compute_smoothness_loss(self, disp, img):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()


# =========================
# 3. 数据集（灰度→3通道 + 成对一致增强）
# =========================
class AugmentedWaveStereoSelfSupervisedDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.is_validation = is_validation
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        if not self.left_images:
            raise FileNotFoundError(f"在路径 '{cfg.LEFT_IMAGE_DIR}' 中没有找到任何图像文件。")
        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42); np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        left_img_path = self.left_images[frame_idx]
        base = os.path.basename(left_img_path)
        right_base = base.replace('lresult', 'rresult', 1) if 'lresult' in base else base
        right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_base)

        left_img  = cv2.imread(left_img_path,  cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            return None

        # 灰度 -> 3通道
        left_img  = np.stack([left_img]*3,  axis=-1)
        right_img = np.stack([right_img]*3, axis=-1)

        # 尺寸策略
        if not self.cfg.PRESERVE_ORIGINAL_RES:
            h, w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
            left_img  = cv2.resize(left_img,  (w, h), interpolation=cv2.INTER_LINEAR)
            right_img = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 成对一致增强（仅训练）
        if (not self.is_validation) and self.cfg.USE_DATA_AUGMENTATION and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
            b = 0.85 + 0.30 * np.random.rand()  # 0.85~1.15
            c = 0.85 + 0.30 * np.random.rand()
            for img in (left_img, right_img):
                img[:] = np.clip(img * b, 0, 255).astype(np.uint8)
                mean = img.mean()
                img[:] = np.clip((img - mean) * c + mean, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.3:
                noise_std = 0.02 * 255 * np.random.rand()
                noise = (noise_std * np.random.randn(*left_img.shape)).astype(np.float32)
                left_img  = np.clip(left_img.astype(np.float32)  + noise, 0, 255).astype(np.uint8)
                right_img = np.clip(right_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        left_tensor  = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0
        return left_tensor, right_tensor


# =========================
# 4. 模型（ViT-B/16 + 分块特征融合）
# =========================
def conv_block_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


class Hourglass3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1a = conv_block_3d(in_channels, in_channels * 2, stride=2)
        self.conv1b = conv_block_3d(in_channels * 2, in_channels * 2)
        self.conv2a = conv_block_3d(in_channels * 2, in_channels * 4, stride=2)
        self.conv2b = conv_block_3d(in_channels * 4, in_channels * 4)
        self.conv3a = conv_block_3d(in_channels * 4, in_channels * 6, stride=2)
        self.conv3b = conv_block_3d(in_channels * 6, in_channels * 6)
        self.deconv3 = conv_block_3d(in_channels * 6, in_channels * 4)
        self.deconv2 = conv_block_3d(in_channels * 4, in_channels * 2)
        self.deconv1 = conv_block_3d(in_channels * 2, in_channels)
        self.redir2 = conv_block_3d(in_channels * 4, in_channels * 4)
        self.redir1 = conv_block_3d(in_channels * 2, in_channels * 2)
        self.redir0 = conv_block_3d(in_channels, in_channels)

    def forward(self, x):
        out1 = self.conv1b(self.conv1a(x))
        out2 = self.conv2b(self.conv2a(out1))
        out3 = self.conv3b(self.conv3a(out2))
        up3 = F.interpolate(out3, size=out2.shape[2:], mode='trilinear', align_corners=False)
        d3  = self.deconv3(up3)
        d2i = F.relu(d3 + self.redir2(out2), inplace=True)
        up2 = F.interpolate(d2i, size=out1.shape[2:], mode='trilinear', align_corners=False)
        d2  = self.deconv2(up2)
        d1i = F.relu(d2 + self.redir1(out1), inplace=True)
        up1 = F.interpolate(d1i, size=x.shape[2:], mode='trilinear', align_corners=False)
        d1  = self.deconv1(up1)
        return F.relu(d1 + self.redir0(x), inplace=True)


class DINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_disp = cfg.MAX_DISPARITY
        self.dino = self._load_dino_model()
        if self.dino is None:
            raise RuntimeError("无法加载DINOv3模型。")
        for p in self.dino.parameters():
            p.requires_grad = False
        self.feature_dim = getattr(self.dino.config, 'hidden_size', 768)
        self.patch_size  = getattr(self.dino.config, 'patch_size', 16)
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 4)

        # 根据成本体积类型选择输入通道数
        in_ch = 1 if cfg.COST_TYPE == "corr" else self.feature_dim
        self.cost_aggregator = nn.Sequential(
            conv_block_3d(in_ch, 32),
            Hourglass3D(32),
            nn.Conv3d(32, 1, 3, padding=1)
        )

        self._last_tiling_sig = None

        print(f"[DINO] loaded: model={cfg.DINO_ONLINE_MODEL}, patch={self.patch_size}, hidden={self.feature_dim}")
        print("模型构建完成 (3D沙漏网络 + 分块特征)。")

    def _load_dino_model(self):
        if os.path.exists(self.cfg.DINO_LOCAL_PATH):
            try:
                return AutoModel.from_pretrained(self.cfg.DINO_LOCAL_PATH, local_files_only=True)
            except Exception as e:
                print(f"[!] 本地加载失败: {e}")
        try:
            model = AutoModel.from_pretrained(self.cfg.DINO_ONLINE_MODEL)
            os.makedirs(self.cfg.DINO_LOCAL_PATH, exist_ok=True)
            model.save_pretrained(self.cfg.DINO_LOCAL_PATH)
            return model
        except Exception as e:
            print(f"[!] 从 Hub 加载失败: {e}")
            return None

    @torch.no_grad()
    def _extract_features_single(self, image_bchw: torch.Tensor) -> torch.Tensor:
        outputs = self.dino(image_bchw)  # [B, 1+reg+N, C]
        features = outputs.last_hidden_state
        start_index = 1 + self.num_register_tokens
        patch_tokens = features[:, start_index:, :]                # [B, N, C]
        b, _, c = patch_tokens.shape
        h, w = image_bchw.shape[-2:]
        ps = self.patch_size
        fh, fw = h // ps, w // ps
        return patch_tokens.permute(0, 2, 1).reshape(b, c, fh, fw)

    @torch.no_grad()
    def _extract_features_tiled(self, image_bchw: torch.Tensor) -> torch.Tensor:
        assert image_bchw.ndim == 4
        b, c, H, W = image_bchw.shape
        ps = self.patch_size

        pad_h = (ps - (H % ps)) % ps
        pad_w = (ps - (W % ps)) % ps
        if self.cfg.PAD_TO_PATCH and (pad_h > 0 or pad_w > 0):
            image_bchw = F.pad(image_bchw, (0, pad_w, 0, pad_h), mode='constant', value=0)
            H += pad_h; W += pad_w

        th_req, tw_req, ov_req = self.cfg.TILE_HEIGHT, self.cfg.TILE_WIDTH, self.cfg.TILE_OVERLAP
        th = max((th_req // ps) * ps, ps)
        tw = max((tw_req // ps) * ps, ps)
        ov = max((ov_req // ps) * ps, 0)
        sig = (th_req, tw_req, ov_req, th, tw, ov, ps)
        if ((th != th_req) or (tw != tw_req) or (ov != ov_req)) and (sig != self._last_tiling_sig):
            print(f"[Tiling] 自动对齐: TILE({th_req},{tw_req})→({th},{tw}), OVERLAP {ov_req}→{ov} (patch={ps})")
            self._last_tiling_sig = sig

        stride_h = max(th - ov, ps)
        stride_w = max(tw - ov, ps)

        fh, fw = H // ps, W // ps
        tile_fh, tile_fw = th // ps, tw // ps

        C = self.feature_dim
        feat_sum   = image_bchw.new_zeros((b, C, fh, fw))
        weight_sum = image_bchw.new_zeros((b, 1, fh, fw))

        win_h = torch.hann_window(tile_fh, periodic=False, device=image_bchw.device).view(1, tile_fh, 1)
        win_w = torch.hann_window(tile_fw, periodic=False, device=image_bchw.device).view(1, 1, tile_fw)
        weight_full = (win_h * win_w).clamp(min=1e-6)  # [1, tile_fh, tile_fw]

        for y in range(0, H, stride_h):
            for x in range(0, W, stride_w):
                y1, x1 = y, x
                y2, x2 = min(y1 + th, H), min(x1 + tw, W)
                tile = image_bchw[:, :, y1:y2, x1:x2]

                cur_fh = (y2 - y1) // ps
                cur_fw = (x2 - x1) // ps

                tile_feats = self._extract_features_single(tile)  # [B,C,cur_fh,cur_fw]
                w2d = weight_full[:, :cur_fh, :cur_fw].unsqueeze(0)  # [1,1,cur_fh,cur_fw]

                yf1, yf2 = y1 // ps, y2 // ps
                xf1, xf2 = x1 // ps, x2 // ps

                feat_sum[:, :, yf1:yf2, xf1:xf2]   += tile_feats * w2d
                weight_sum[:, :, yf1:yf2, xf1:xf2] += w2d

        features_2d = feat_sum / weight_sum.clamp(min=1e-6)

        if self.cfg.PAD_TO_PATCH and (pad_h > 0 or pad_w > 0):
            fh_org = (H - pad_h) // ps
            fw_org = (W - pad_w) // ps
            features_2d = features_2d[:, :, :fh_org, :fw_org]

        return features_2d

    def get_features(self, image_bchw: torch.Tensor) -> torch.Tensor:
        if self.cfg.USE_TILING:
            return self._extract_features_tiled(image_bchw)
        else:
            b, c, H, W = image_bchw.shape
            ps = self.patch_size
            pad_h = (ps - (H % ps)) % ps
            pad_w = (ps - (W % ps)) % ps
            if (pad_h > 0 or pad_w > 0):
                image_bchw = F.pad(image_bchw, (0, pad_w, 0, pad_h), mode='constant', value=0)
            feats = self._extract_features_single(image_bchw)
            if (pad_h > 0 or pad_w > 0):
                fh_org = H // ps; fw_org = W // ps
                feats = feats[:, :, :fh_org, :fw_org]
            return feats

    def build_cost_volume(self, left_feat, right_feat):
        B, C, H, W = left_feat.shape
        max_disp_feat = max(1, self.max_disp // self.patch_size)

        if self.cfg.COST_TYPE == "corr":
            # 相关性体积：通道内点积 → 单通道（取负以配合 softmax(-cost)）
            cost_volume = left_feat.new_zeros(B, 1, max_disp_feat, H, W)
            for d in range(max_disp_feat):
                if d > 0:
                    corr = (left_feat[:, :, :, d:] * right_feat[:, :, :, :-d]).sum(dim=1, keepdim=True)
                    cost_volume[:, :, d, :, d:] = -corr
                else:
                    corr = (left_feat * right_feat).sum(dim=1, keepdim=True)
                    cost_volume[:, :, d, :, :] = -corr
            return cost_volume
        else:
            # 差分体积：按通道差分
            cost_volume = left_feat.new_zeros(B, C, max_disp_feat, H, W)
            for d in range(max_disp_feat):
                if d > 0:
                    cost_volume[:, :, d, :, d:] = left_feat[:, :, :, d:] - right_feat[:, :, :, :-d]
                else:
                    cost_volume[:, :, d, :, :] = left_feat - right_feat
            return cost_volume

    def forward(self, left_image, right_image):
        h, w = left_image.shape[-2:]
        left_feat  = self.get_features(left_image)
        right_feat = self.get_features(right_image)

        # ★ 通道维归一化（显著稳定相似性度量）
        left_feat  = F.normalize(left_feat,  dim=1, eps=1e-6)
        right_feat = F.normalize(right_feat, dim=1, eps=1e-6)

        cost_volume = self.build_cost_volume(left_feat, right_feat)           # [B,C'|1,D,Hf,Wf]
        cost_agg    = self.cost_aggregator(cost_volume).squeeze(1)           # [B,D,Hf,Wf]
        prob = F.softmax(-cost_agg, dim=1)
        max_disp_feat = max(1, self.max_disp // self.patch_size)
        disp_values = torch.arange(0, max_disp_feat, device=prob.device, dtype=torch.float32).view(1, -1, 1, 1)
        disp_feat = torch.sum(prob * disp_values, 1, keepdim=True)           # [B,1,Hf,Wf]
        disparity = F.interpolate(disp_feat * self.patch_size, size=(h, w), mode='bilinear', align_corners=False)

        cap_threshold = (self.max_disp - self.patch_size)
        with torch.no_grad():
            cap_ratio = (disparity >= cap_threshold).float().mean().item()

        disparity_right = None
        if self.cfg.USE_CONSISTENCY_LOSS:
            cost_volume_r = self.build_cost_volume(right_feat, left_feat)
            cost_agg_r    = self.cost_aggregator(cost_volume_r).squeeze(1)
            prob_r        = F.softmax(-cost_agg_r, dim=1)
            disp_feat_r   = torch.sum(prob_r * disp_values, 1, keepdim=True)
            disparity_right = F.interpolate(disp_feat_r * self.patch_size, size=(h, w), mode='bilinear', align_corners=False)

        return {"disparity": disparity, "disparity_right": disparity_right, "cap_ratio": cap_ratio}


# =========================
# 5. 评价指标
# =========================
class EvaluationMetrics:
    @staticmethod
    def compute_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0: return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    @staticmethod
    def compute_rmse(img1, img2):
        return torch.sqrt(torch.mean((img1 - img2) ** 2)).item()

    @staticmethod
    def compute_ssim(img1, img2):
        ssim_module = SSIM()
        return (1 - 2 * ssim_module(img1, img2).mean()).item()

    @staticmethod
    def evaluate_reconstruction(inputs, outputs, loss_components):
        left_img = inputs["left_image"]
        warped_right = loss_components["warped_right_image"]
        return {
            "psnr": EvaluationMetrics.compute_psnr(left_img, warped_right),
            "rmse": EvaluationMetrics.compute_rmse(left_img, warped_right),
            "ssim": EvaluationMetrics.compute_ssim(left_img, warped_right),
        }


# =========================
# 6. 训练器
# =========================
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    return torch.utils.data.dataloader.default_collate(batch)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        os.makedirs(cfg.TENSORBOARD_DIR, exist_ok=True)
        if cfg.VISUALIZE_TRAINING: os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu':
            print("【警告】未检测到 GPU，将使用 CPU，训练会很慢。")
            cfg.USE_MIXED_PRECISION = False
        print(f"✓ 使用设备: {self.device}")

        self.writer = SummaryWriter(log_dir=cfg.TENSORBOARD_DIR) if SummaryWriter is not None else None

        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.loss_fn = ImprovedSelfSupervisedLoss(
            smoothness_weight=cfg.INITIAL_SMOOTHNESS_WEIGHT,
            photometric_weights=cfg.PHOTOMETRIC_LOSS_WEIGHTS,
            use_consistency_loss=cfg.USE_CONSISTENCY_LOSS,
            consistency_weight=cfg.CONSISTENCY_LOSS_WEIGHT
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)
        self.evaluator = EvaluationMetrics()

        train_dataset = AugmentedWaveStereoSelfSupervisedDataset(cfg, is_validation=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                                       num_workers=4, pin_memory=self.device.type == 'cuda')
        val_dataset = AugmentedWaveStereoSelfSupervisedDataset(cfg, is_validation=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=4, pin_memory=self.device.type == 'cuda')

        self.scaler = _amp.GradScaler('cuda', enabled=cfg.USE_MIXED_PRECISION)
        self.step = 0
        self.current_smoothness_weight = cfg.INITIAL_SMOOTHNESS_WEIGHT

        self.loss_history = {
            'train': {'total': [], 'photometric': [], 'smoothness': [], 'consistency': []},
            'val':   {'total': [], 'photometric': [], 'smoothness': [], 'consistency': []}
        }
        self.metric_history = {
            'train': {'psnr': [], 'rmse': [], 'ssim': [], 'cap_ratio': []},
            'val':   {'psnr': [], 'rmse': [], 'ssim': [], 'cap_ratio': []}
        }

        self._setup_visualization_font()
        self.log_file = os.path.join(cfg.LOG_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def _setup_visualization_font(self):
        font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'sans-serif']
        for font_name in font_names:
            try:
                if any(font.name == font_name for font in fm.fontManager.ttflist):
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"✓ 可视化已配置中文字体: {font_name}")
                    return
            except Exception:
                continue
        print("警告: 未找到中文字体，将使用默认字体。")

    def train(self):
        print("--- 开始自监督训练 ---")
        best_val_loss = float('inf')

        for epoch in range(self.cfg.NUM_EPOCHS):
            self.current_smoothness_weight *= self.cfg.SMOOTHNESS_WEIGHT_DECAY
            self.loss_fn.smoothness_weight = self.current_smoothness_weight

            self.model.train()
            epoch_train_loss_total = 0.0
            epoch_train_metrics = {'psnr': 0.0, 'rmse': 0.0, 'ssim': 0.0, 'cap_ratio': 0.0}
            last_train_loss_components = {}

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [训练]")

            for data in pbar:
                if data is None or data[0] is None: continue
                left, right = [d.to(self.device) for d in data]

                self.optimizer.zero_grad()
                with _amp.autocast('cuda', enabled=self.cfg.USE_MIXED_PRECISION):
                    outputs = self.model(left, right)
                    inputs = {"left_image": left, "right_image": right}
                    loss_components = self.loss_fn(inputs, outputs)
                    loss = loss_components["total_loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 训练步骤 {self.step} 出现无效损失，跳过。")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                step_metrics = self.evaluator.evaluate_reconstruction(inputs, outputs, loss_components)
                step_metrics['cap_ratio'] = outputs.get("cap_ratio", 0.0)

                epoch_train_loss_total += loss.item()
                for k in epoch_train_metrics:
                    epoch_train_metrics[k] += step_metrics.get(k, 0.0)
                last_train_loss_components = loss_components

                pbar.set_postfix({'loss': loss.item(),
                                  'psnr': f"{step_metrics['psnr']:.2f}",
                                  'cap':  f"{step_metrics['cap_ratio']:.3f}",
                                  'lr':   f"{self.optimizer.param_groups[0]['lr']:.1e}"})

                if self.writer is not None:
                    self.writer.add_scalar('Loss/train_step', loss.item(), self.step)
                    self.writer.add_scalar('Metrics/train_psnr_step', step_metrics['psnr'], self.step)
                    self.writer.add_scalar('Metrics/train_cap_ratio_step', step_metrics['cap_ratio'], self.step)
                    self.writer.add_scalar('Params/learning_rate', self.optimizer.param_groups[0]['lr'], self.step)

                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(inputs, outputs, loss_components, self.step, "train")

                self.step += 1

            train_len = len(self.train_loader)
            if train_len > 0:
                avg_train_loss = epoch_train_loss_total / train_len
                avg_train_metrics = {k: v / train_len for k, v in epoch_train_metrics.items()}

                self.loss_history['train']['total'].append(avg_train_loss)
                self.loss_history['train']['photometric'].append(last_train_loss_components.get('photometric_loss', torch.tensor(0)).item())
                self.loss_history['train']['smoothness'].append(last_train_loss_components.get('smoothness_loss', torch.tensor(0)).item())
                self.loss_history['train']['consistency'].append(last_train_loss_components.get('consistency_loss', torch.tensor(0)).item())
                for k, v in avg_train_metrics.items():
                    self.metric_history['train'][k].append(v)

                if self.writer is not None:
                    self.writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
                    self.writer.add_scalar('Metrics/train_psnr_epoch', avg_train_metrics['psnr'], epoch)
                    self.writer.add_scalar('Metrics/train_cap_ratio_epoch', avg_train_metrics['cap_ratio'], epoch)

            avg_val_loss, val_metrics = self.validate(epoch)

            # —— 改进的控制台打印：包含损失分解 —— #
            ph = self.loss_history['val']['photometric'][-1] if self.loss_history['val']['photometric'] else float('nan')
            sm = self.loss_history['val']['smoothness'][-1]  if self.loss_history['val']['smoothness']  else float('nan')
            cs = self.loss_history['val']['consistency'][-1] if self.loss_history['val']['consistency'] else float('nan')
            print(
                f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> "
                f"验证损失: {avg_val_loss:.4f}, "
                f"PSNR: {val_metrics.get('psnr', 0):.2f}, "
                f"cap: {val_metrics.get('cap_ratio', 0):.3f}, "
                f"photo: {ph:.4f}, smooth: {sm:.4f}, cons: {cs:.4f}"
            )

            self.scheduler.step(avg_val_loss)
            self.update_log_file(epoch)

            if self.cfg.VISUALIZE_TRAINING:
                self.plot_training_history()

            if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model_self_supervised.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ 验证损失降低，模型已保存至: {save_path}")

        print("训练完成!")
        if self.writer is not None:
            self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        total_loss, val_psnr, val_ssim, val_rmse, val_cap = 0, 0, 0, 0, 0
        last_val_loss_components = {}

        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="[验证]"):
                if data is None or data[0] is None: continue
                left, right = [d.to(self.device) for d in data]
                outputs = self.model(left, right)
                inputs = {"left_image": left, "right_image": right}
                loss_components = self.loss_fn(inputs, outputs)
                last_val_loss_components = loss_components

                if not torch.isnan(loss_components["total_loss"]):
                    total_loss += loss_components["total_loss"].item()

                step_metrics = self.evaluator.evaluate_reconstruction(inputs, outputs, loss_components)
                val_psnr += step_metrics['psnr']
                val_ssim += step_metrics['ssim']
                val_rmse += step_metrics['rmse']
                val_cap  += outputs.get("cap_ratio", 0.0)

        val_len = len(self.val_loader)
        avg_loss = total_loss / val_len if val_len > 0 else 0
        avg_metrics = {'psnr': val_psnr / val_len,
                       'ssim': val_ssim / val_len,
                       'rmse': val_rmse / val_len,
                       'cap_ratio': val_cap / val_len} if val_len > 0 else {}

        if val_len > 0:
            self.loss_history['val']['total'].append(avg_loss)
            self.loss_history['val']['photometric'].append(last_val_loss_components.get('photometric_loss', torch.tensor(0)).item())
            self.loss_history['val']['smoothness'].append(last_val_loss_components.get('smoothness_loss', torch.tensor(0)).item())
            self.loss_history['val']['consistency'].append(last_val_loss_components.get('consistency_loss', torch.tensor(0)).item())
            for k, v in avg_metrics.items():
                self.metric_history['val'][k].append(v)

            if self.writer is not None and self.loss_history['train']['total']:
                self.writer.add_scalar('Loss/validation_epoch', avg_loss, epoch)
                self.writer.add_scalar('Metrics/validation_psnr_epoch', avg_metrics['psnr'], epoch)
                self.writer.add_scalar('Metrics/validation_ssim_epoch', avg_metrics['ssim'], epoch)
                self.writer.add_scalar('Metrics/validation_rmse_epoch', avg_metrics['rmse'], epoch)
                self.writer.add_scalar('Metrics/validation_cap_ratio_epoch', avg_metrics['cap_ratio'], epoch)
                self.writer.add_scalars('Loss/epoch_comparison',
                                        {'train': self.loss_history['train']['total'][-1], 'validation': avg_loss},
                                        epoch)

        return avg_loss, avg_metrics

    def visualize(self, inputs, outputs, loss_components, step, phase="train"):
        left_img  = inputs["left_image"][0].permute(1, 2, 0).cpu().numpy()
        right_img = inputs["right_image"][0].permute(1, 2, 0).cpu().numpy()
        pred_disp = outputs["disparity"][0, 0].cpu().detach().numpy()

        disp_min, disp_max, disp_mean, disp_std = pred_disp.min(), pred_disp.max(), pred_disp.mean(), pred_disp.std()

        fig = plt.figure(figsize=(20, 15))
        plt.suptitle(f'可视化 - 步骤: {step} ({phase})', fontsize=16)

        plt.subplot(3, 4, 1); plt.imshow(left_img);  plt.title("左图"); plt.axis('off')
        plt.subplot(3, 4, 2); plt.imshow(right_img); plt.title("右图"); plt.axis('off')

        ax3 = plt.subplot(3, 4, 3)
        im3 = ax3.imshow(pred_disp, cmap='viridis')
        plt.title(f"预测视差 (min:{disp_min:.2f}, max:{disp_max:.2f})"); plt.axis('off')
        fig.colorbar(im3, ax=ax3)

        plt.subplot(3, 4, 4); plt.hist(pred_disp.flatten(), bins=50); plt.title(f"视差分布 (μ:{disp_mean:.2f}, σ:{disp_std:.2f})")

        warped_right = loss_components["warped_right_image"][0].permute(1, 2, 0).cpu().detach().numpy()
        plt.subplot(3, 4, 5); plt.imshow(warped_right); plt.title("重建的左图"); plt.axis('off')

        diff_right = np.clip(np.abs(left_img - warped_right), 0, 1)
        ax6 = plt.subplot(3, 4, 6)
        im6 = ax6.imshow(diff_right, cmap='hot')
        plt.title("重建误差图 (左)"); plt.axis('off')
        fig.colorbar(im6, ax=ax6)

        if "warped_left_image" in loss_components:
            warped_left = loss_components["warped_left_image"][0].permute(1, 2, 0).cpu().detach().numpy()
            diff_left = np.clip(np.abs(right_img - warped_left), 0, 1)
            plt.subplot(3, 4, 7); plt.imshow(warped_left); plt.title("重建的右图"); plt.axis('off')
            ax8 = plt.subplot(3, 4, 8)
            im8 = ax8.imshow(diff_left, cmap='hot')
            plt.title("重建误差图 (右)"); plt.axis('off')
            fig.colorbar(im8, ax=ax8)

        info_text = (
            f"总损失: {loss_components['total_loss'].item():.4f}\n"
            f"光度损失: {loss_components['photometric_loss'].item():.4f}\n"
            f"平滑损失: {loss_components['smoothness_loss'].item():.4f}\n"
            f"一致性损失: {loss_components.get('consistency_loss', torch.tensor(0)).item():.4f}\n"
            f"cap_ratio: {outputs.get('cap_ratio', 0.0):.3f}\n"
        )
        plt.subplot(3, 4, 9, frameon=False); plt.xticks([]); plt.yticks([])
        plt.text(0.1, 0.5, info_text, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, f"{phase}_step_{step:06d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if self.writer is not None:
            self.writer.add_image(f'Images/{phase}_left', torch.from_numpy(left_img).permute(2, 0, 1), global_step=step)
            self.writer.add_image(f'Images/{phase}_disparity', torch.from_numpy(pred_disp).unsqueeze(0), global_step=step)
            self.writer.add_image(f'Images/{phase}_warped_left', torch.from_numpy(warped_right).permute(2, 0, 1), global_step=step)

    def plot_training_history(self):
        if not self.loss_history['train']['total']:
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练历史曲线', fontsize=16)

        axes[0, 0].plot(self.loss_history['train']['total'], label='训练损失')
        if self.loss_history['val']['total']:
            axes[0, 0].plot(self.loss_history['val']['total'], label='验证损失')
        axes[0, 0].set_title('总损失'); axes[0, 0].legend(); axes[0, 0].grid(True)

        axes[0, 1].plot(self.loss_history['train']['photometric'], label='训练光度损失')
        axes[0, 1].plot(self.loss_history['train']['smoothness'], label='训练平滑损失')
        if self.loss_history['val']['photometric']:
            axes[0, 1].plot(self.loss_history['val']['photometric'], label='验证光度损失')
        if self.loss_history['val']['smoothness']:
            axes[0, 1].plot(self.loss_history['val']['smoothness'], label='验证平滑损失')
        axes[0, 1].set_title('损失组件'); axes[0, 1].legend(); axes[0, 1].grid(True)

        if self.metric_history['train']['psnr']:
            axes[1, 0].plot(self.metric_history['train']['psnr'], label='训练PSNR')
        if self.metric_history['val']['psnr']:
            axes[1, 0].plot(self.metric_history['val']['psnr'], label='验证PSNR')
        axes[1, 0].set_title('PSNR'); axes[1, 0].legend(); axes[1, 0].grid(True)

        if self.metric_history['train']['ssim']:
            axes[1, 1].plot(self.metric_history['train']['ssim'], label='训练SSIM')
        if self.metric_history['val']['ssim']:
            axes[1, 1].plot(self.metric_history['val']['ssim'], label='验证SSIM')
        axes[1, 1].set_title('SSIM'); axes[1, 1].legend(); axes[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, "training_history.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def update_log_file(self, epoch):
        log_data = {
            'config': asdict(self.cfg),
            'epoch': epoch,
            'loss_history': self.loss_history,
            'metric_history': self.metric_history,
            'current_smoothness_weight': self.current_smoothness_weight,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'update_time': datetime.now().isoformat()
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)


# =========================
# 7. 主入口（分目录运行）
# =========================
def build_run_tag(cfg: Config) -> str:
    if cfg.RUN_TAG:
        return cfg.RUN_TAG
    tag = f"vitb16_origres_{'tile' if cfg.USE_TILING else 'noTile'}" \
          f"_{cfg.TILE_HEIGHT}x{cfg.TILE_WIDTH}_ov{cfg.TILE_OVERLAP}_md{cfg.MAX_DISPARITY}"
    return tag + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == '__main__':
    plt.switch_backend('Agg')

    config = Config()
    run_tag = build_run_tag(config)
    base_dir = os.path.join("./experiments/", run_tag)
    config.CHECKPOINT_DIR   = os.path.join(base_dir, "checkpoints/")
    config.VISUALIZATION_DIR= os.path.join(base_dir, "visualization/")
    config.LOG_DIR          = os.path.join(base_dir, "logs/")
    config.TENSORBOARD_DIR  = os.path.join(base_dir, "runs/")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)

    print(f"[Run] tag={run_tag}")
    print(f"[Dirs] ckpt={config.CHECKPOINT_DIR}")
    print(f"[Dirs] vis ={config.VISUALIZATION_DIR}")
    print(f"[Dirs] log ={config.LOG_DIR}")
    print(f"[Dirs] tb  ={config.TENSORBOARD_DIR}")

    trainer = Trainer(config)
    trainer.train()
