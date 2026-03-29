# self_supervised_dinov3.py
# 实现了自监督的、端到端的波浪表面三维重建方案。
# (V5 最终性能版：引入3D沙漏网络进行代价聚合，旨在突破性能瓶颈)

import os
import sys
import glob
from dataclasses import dataclass
import json
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

# --- 使用AutoModel以智能加载DINOv3 ---
try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80)
    print("【致命错误】: 无法从 'transformers' 库中导入 'AutoModel'。")
    # ... (保留错误信息)
    sys.exit(1)


# --- 1. 配置中心 ---
@dataclass
class Config:
    """项目配置参数"""
    # --- 文件路径 ---
    LEFT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/lresult/"
    RIGHT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/rresult/"
    CALIBRATION_FILE: str = "D:/Research/wave_reconstruction_project/camera_calibration/params/stereo_calib_params_from_matlab_full.npz"

    # --- 自动生成的数据路径 ---
    CHECKPOINT_DIR: str = "./checkpoints_self_supervised/"
    VISUALIZATION_DIR: str = "D:/Research/wave_reconstruction_project/data/visualization_self_supervised/"
    LOG_DIR: str = "./logs/"

    # --- DINOv3 模型配置 ---
    DINO_ONLINE_MODEL: str = "facebook/dinov3_vitb14"
    DINO_LOCAL_PATH: str = "./dinov3-base-model/"

    # --- 可视化控制开关 ---
    VISUALIZE_TRAINING: bool = True
    VISUALIZE_INTERVAL: int = 100

    # --- 数据处理参数 ---
    IMAGE_HEIGHT: int = 252
    IMAGE_WIDTH: int = 504

    # --- 模型与训练参数 ---
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1
    SMOOTHNESS_LOSS_WEIGHT: float = 0.1
    GRADIENT_CLIP_VAL: float = 1.0
    # [核心修改] 新增最大视差参数，用于构建代价空间
    MAX_DISPARITY: int = 126  # 通常设置为宽度的1/4，且为patch size的倍数

    # --- 新增优化参数 ---
    USE_MIXED_PRECISION: bool = True  # 使用混合精度训练
    USE_DATA_AUGMENTATION: bool = True  # 使用数据增强
    AUGMENTATION_PROBABILITY: float = 0.5  # 数据增强概率
    PHOTOMETRIC_LOSS_WEIGHTS: tuple = (0.85, 0.15)  # (SSIM权重, L1权重)
    USE_CONSISTENCY_LOSS: bool = True  # 使用左右一致性损失
    CONSISTENCY_LOSS_WEIGHT: float = 0.1  # 一致性损失权重
    INITIAL_SMOOTHNESS_WEIGHT: float = 0.1  # 初始平滑损失权重
    SMOOTHNESS_WEIGHT_DECAY: float = 0.95  # 每epoch平滑损失权重衰减


# --- 2. 自监督损失函数 (L1+SSIM) ---
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
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
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class ImprovedSelfSupervisedLoss(nn.Module):
    def __init__(self, smoothness_weight=0.1, photometric_weights=(0.85, 0.15),
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

        # 双向 warp 损失
        warped_right_image = self.inverse_warp(right_img, pred_disp)
        warped_left_image = self.inverse_warp(left_img, -pred_disp)  # 反向视差

        # 双向光度损失
        l1_loss_right = torch.abs(warped_right_image - left_img).mean()
        l1_loss_left = torch.abs(warped_left_image - right_img).mean()
        l1_loss = (l1_loss_right + l1_loss_left) / 2

        ssim_loss_right = self.ssim(warped_right_image, left_img).mean()
        ssim_loss_left = self.ssim(warped_left_image, right_img).mean()
        ssim_loss = (ssim_loss_right + ssim_loss_left) / 2

        photometric_loss = self.photometric_weights[0] * ssim_loss + self.photometric_weights[1] * l1_loss

        # 平滑损失
        smoothness_loss = self.compute_smoothness_loss(pred_disp, left_img)

        # 左右一致性损失
        consistency_loss = 0
        if self.use_consistency_loss and "disparity_right" in outputs:
            disp_left = outputs["disparity"]
            disp_right = outputs["disparity_right"]
            # 将右视差图warp到左视图
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
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=0).float().to(features.device)
        pixel_coords = pixel_coords.repeat(B, 1, 1, 1)
        disp = disp.squeeze(1)
        transformed_x = pixel_coords[:, 0, :, :] - disp
        grid = torch.stack([transformed_x, pixel_coords[:, 1, :, :]], dim=-1)
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def compute_smoothness_loss(self, disp, img):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()


# --- 3. PyTorch 数据集 ---
class AugmentedWaveStereoSelfSupervisedDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.is_validation = is_validation
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        left_img_path = self.left_images[frame_idx]
        frame_basename = os.path.basename(left_img_path)
        right_frame_basename = frame_basename.replace('lresult', 'rresult', 1)
        right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_frame_basename)
        try:
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)
            if left_img is None or right_img is None: return None
        except Exception:
            return None
        target_h, target_w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
        left_img = cv2.resize(left_img, (target_w, target_h))
        right_img = cv2.resize(right_img, (target_w, target_h))

        # 数据增强 (仅在训练时使用)
        if not self.is_validation and self.cfg.USE_DATA_AUGMENTATION and np.random.rand() < self.cfg.AUGMENTATION_PROBABILITY:
            # 随机亮度调整
            brightness_factor = 0.8 + 0.4 * np.random.rand()
            left_img = np.clip(left_img * brightness_factor, 0, 255).astype(np.uint8)
            right_img = np.clip(right_img * brightness_factor, 0, 255).astype(np.uint8)

            # 随机对比度调整
            contrast_factor = 0.8 + 0.4 * np.random.rand()
            left_mean = left_img.mean()
            right_mean = right_img.mean()
            left_img = np.clip((left_img - left_mean) * contrast_factor + left_mean, 0, 255).astype(np.uint8)
            right_img = np.clip((right_img - right_mean) * contrast_factor + right_mean, 0, 255).astype(np.uint8)

            # 随机高斯噪声
            if np.random.rand() < 0.3:
                noise_std = 0.02 * 255 * np.random.rand()
                left_img = np.clip(left_img + noise_std * np.random.randn(*left_img.shape), 0, 255).astype(np.uint8)
                right_img = np.clip(right_img + noise_std * np.random.randn(*right_img.shape), 0, 255).astype(np.uint8)

        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0

        return left_tensor, right_tensor


# --- 4. 深度学习模型 (核心架构重构) ---

# [核心修改] 为沙漏网络定义基本的3D卷积块
def conv_block_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


# [核心修改] 定义3D沙漏网络模块
class Hourglass3D(nn.Module):
    def __init__(self, in_channels):
        super(Hourglass3D, self).__init__()

        # Encoder path
        self.conv1a = conv_block_3d(in_channels, in_channels * 2, stride=2)
        self.conv1b = conv_block_3d(in_channels * 2, in_channels * 2)

        self.conv2a = conv_block_3d(in_channels * 2, in_channels * 4, stride=2)
        self.conv2b = conv_block_3d(in_channels * 4, in_channels * 4)

        self.conv3a = conv_block_3d(in_channels * 4, in_channels * 6, stride=2)
        self.conv3b = conv_block_3d(in_channels * 6, in_channels * 6)

        # Decoder path (using regular convolutions after upsampling)
        self.deconv3 = conv_block_3d(in_channels * 6, in_channels * 4)
        self.deconv2 = conv_block_3d(in_channels * 4, in_channels * 2)
        self.deconv1 = conv_block_3d(in_channels * 2, in_channels)

        # Skip connection refinement layers
        self.redir2 = conv_block_3d(in_channels * 4, in_channels * 4)
        self.redir1 = conv_block_3d(in_channels * 2, in_channels * 2)
        self.redir0 = conv_block_3d(in_channels, in_channels)

    def forward(self, x):
        # --- Encoder ---
        out_conv1 = self.conv1b(self.conv1a(x))
        out_conv2 = self.conv2b(self.conv2a(out_conv1))
        out_conv3 = self.conv3b(self.conv3a(out_conv2))

        # --- Decoder with fixed upsampling ---
        # Upsample from Level 3 to Level 2
        up3 = F.interpolate(out_conv3, size=out_conv2.shape[2:], mode='trilinear', align_corners=False)
        deconv3_out = self.deconv3(up3)
        deconv2_in = F.relu(deconv3_out + self.redir2(out_conv2), inplace=True)

        # Upsample from Level 2 to Level 1
        up2 = F.interpolate(deconv2_in, size=out_conv1.shape[2:], mode='trilinear', align_corners=False)
        deconv2_out = self.deconv2(up2)
        deconv1_in = F.relu(deconv2_out + self.redir1(out_conv1), inplace=True)

        # Upsample from Level 1 to Level 0
        up1 = F.interpolate(deconv1_in, size=x.shape[2:], mode='trilinear', align_corners=False)
        deconv1_out = self.deconv1(up1)
        final_out = F.relu(deconv1_out + self.redir0(x), inplace=True)

        return final_out


class DINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.max_disp = cfg.MAX_DISPARITY

        self.dino = self._load_dino_model()
        if self.dino is None: raise RuntimeError("无法加载DINOv3模型。")

        for param in self.dino.parameters():
            param.requires_grad = False

        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 4)

        # [核心修改] 使用沙漏网络作为代价聚合器
        self.cost_aggregator = nn.Sequential(
            conv_block_3d(self.feature_dim, 32),
            Hourglass3D(32),
            nn.Conv3d(32, 1, 3, padding=1)  # 最后输出单通道的成本
        )
        print("模型构建完成 (3D沙漏网络架构)。")

    def _load_dino_model(self):
        if os.path.exists(self.cfg.DINO_LOCAL_PATH):
            try:
                return AutoModel.from_pretrained(self.cfg.DINO_LOCAL_PATH, local_files_only=True)
            except Exception as e:
                print(f"[!] 从本地加载模型失败: {e}。")
        try:
            model = AutoModel.from_pretrained(self.cfg.DINO_ONLINE_MODEL)
            model.save_pretrained(self.cfg.DINO_LOCAL_PATH)
            return model
        except Exception as e:
            print(f"[!] 从Hugging Face Hub加载模型失败: {e}")
            return None

    def get_features(self, image):
        b, c, h, w = image.shape
        with torch.no_grad():
            outputs = self.dino(image)
            features = outputs.last_hidden_state
        start_index = 1 + self.num_register_tokens
        patch_tokens = features[:, start_index:, :]
        feature_h, feature_w = h // self.patch_size, w // self.patch_size
        features_2d = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, feature_h, feature_w)
        return features_2d

    def build_cost_volume(self, left_feat, right_feat):
        B, C, H, W = left_feat.shape
        max_disp_feat = self.max_disp // self.patch_size
        cost_volume = torch.zeros(B, C, max_disp_feat, H, W).to(left_feat.device)

        for d in range(max_disp_feat):
            if d > 0:
                cost_volume[:, :, d, :, d:] = left_feat[:, :, :, d:] - right_feat[:, :, :, :-d]
            else:
                cost_volume[:, :, d, :, :] = left_feat - right_feat

        return cost_volume

    def forward(self, left_image, right_image):
        h, w = left_image.shape[-2:]
        left_feat = self.get_features(left_image)
        right_feat = self.get_features(right_image)
        cost_volume = self.build_cost_volume(left_feat, right_feat)
        cost_aggregated = self.cost_aggregator(cost_volume).squeeze(1)
        cost_softmax = F.softmax(-cost_aggregated, dim=1)

        disp_values = torch.arange(0, self.max_disp // self.patch_size, device=cost_softmax.device, dtype=torch.float32)
        disp_values = disp_values.view(1, -1, 1, 1)

        disparity_feat = torch.sum(cost_softmax * disp_values, 1, keepdim=True)
        disparity = F.interpolate(disparity_feat * self.patch_size, size=(h, w), mode='bilinear', align_corners=False)

        # 如果需要计算一致性损失，也计算右视差图
        disparity_right = None
        if self.cfg.USE_CONSISTENCY_LOSS:
            # 交换左右图像计算右视差图
            right_feat_for_left = self.get_features(right_image)
            left_feat_for_right = self.get_features(left_image)
            cost_volume_right = self.build_cost_volume(right_feat_for_left, left_feat_for_right)
            cost_aggregated_right = self.cost_aggregator(cost_volume_right).squeeze(1)
            cost_softmax_right = F.softmax(-cost_aggregated_right, dim=1)

            disparity_feat_right = torch.sum(cost_softmax_right * disp_values, 1, keepdim=True)
            disparity_right = F.interpolate(disparity_feat_right * self.patch_size, size=(h, w), mode='bilinear',
                                            align_corners=False)

        return {"disparity": disparity, "disparity_right": disparity_right}


# --- 5. 评价指标 ---
class EvaluationMetrics:
    """用于评估模型性能的指标计算"""

    @staticmethod
    def compute_psnr(img1, img2):
        """计算PSNR指标"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    @staticmethod
    def compute_rmse(img1, img2):
        """计算RMSE指标"""
        return torch.sqrt(torch.mean((img1 - img2) ** 2)).item()

    @staticmethod
    def compute_ssim(img1, img2):
        """计算SSIM指标"""
        ssim_module = SSIM()
        return (1 - 2 * ssim_module(img1, img2).mean()).item()  # 转换为相似度而非距离

    @staticmethod
    def evaluate_reconstruction(inputs, outputs, loss_components):
        """评估重建质量"""
        left_img = inputs["left_image"]
        warped_right = loss_components["warped_right_image"]

        metrics = {
            "psnr": EvaluationMetrics.compute_psnr(left_img, warped_right),
            "rmse": EvaluationMetrics.compute_rmse(left_img, warped_right),
            "ssim": EvaluationMetrics.compute_ssim(left_img, warped_right),
        }

        return metrics


# --- 6. 训练器 ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    return torch.utils.data.dataloader.default_collate(batch)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        if cfg.VISUALIZE_TRAINING: os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)
        if not torch.cuda.is_available(): sys.exit("【致命错误】: 未检测到GPU。")
        self.device = torch.device("cuda")
        print(f"✓ 成功检测到GPU，将使用设备: {self.device}")

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
                                       num_workers=4, pin_memory=True)
        val_dataset = AugmentedWaveStereoSelfSupervisedDataset(cfg, is_validation=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=4, pin_memory=True)

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_MIXED_PRECISION)

        self.step = 0
        self.current_smoothness_weight = cfg.INITIAL_SMOOTHNESS_WEIGHT

        # 记录训练历史
        self.loss_history = {
            'train': {'total': [], 'photo': [], 'smooth': [], 'consistency': []},
            'val': {'total': [], 'photo': [], 'smooth': [], 'consistency': []}
        }
        self.metric_history = {
            'train': {'psnr': [], 'rmse': [], 'ssim': []},
            'val': {'psnr': [], 'rmse': [], 'ssim': []}
        }

        self._setup_visualization_font()

        # 创建日志文件
        self.log_file = os.path.join(cfg.LOG_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(self.log_file, 'w') as f:
            json.dump({
                'config': cfg.__dict__,
                'start_time': datetime.now().isoformat(),
                'loss_history': self.loss_history,
                'metric_history': self.metric_history
            }, f, indent=2)

    def _setup_visualization_font(self):
        font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'sans-serif']
        for font_name in font_names:
            try:
                if any(font.name == font_name for font in fm.fontManager.ttflist):
                    plt.rcParams['font.sans-serif'] = [font_name];
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"✓ 可视化已配置中文字体: {font_name}")
                    return
            except Exception:
                continue
        print("警告: 未找到指定的中文字体。")

    def train(self):
        print("--- 开始自监督训练 ---")
        best_val_loss = float('inf')

        for epoch in range(self.cfg.NUM_EPOCHS):
            # 更新平滑损失权重
            self.current_smoothness_weight *= self.cfg.SMOOTHNESS_WEIGHT_DECAY
            self.loss_fn.smoothness_weight = self.current_smoothness_weight

            # 训练阶段
            self.model.train()
            train_metrics = {'psnr': 0, 'rmse': 0, 'ssim': 0}
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [训练]")

            for data in pbar:
                if data is None or data[0] is None: continue
                left, right = data
                left, right = left.to(self.device), right.to(self.device)

                self.optimizer.zero_grad()

                # 混合精度前向传播
                with torch.cuda.amp.autocast(enabled=self.cfg.USE_MIXED_PRECISION):
                    outputs = self.model(left, right)
                    inputs = {"left_image": left, "right_image": right}
                    loss_components = self.loss_fn(inputs, outputs)
                    loss = loss_components["total_loss"]

                # 混合精度反向传播
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 计算评估指标
                step_metrics = self.evaluator.evaluate_reconstruction(inputs, outputs, loss_components)
                for k in train_metrics:
                    train_metrics[k] += step_metrics[k]

                pbar.set_postfix({
                    'loss': loss.item(),
                    'psnr': step_metrics['psnr'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })

                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(inputs, outputs, loss_components, self.step, "train")

                self.step += 1

            # 记录训练损失和指标
            for k in self.loss_history['train']:
                self.loss_history['train'][k].append(loss_components[k].item() if k in loss_components else 0)

            for k in train_metrics:
                self.metric_history['train'][k].append(train_metrics[k] / len(self.train_loader))

            # 验证阶段
            avg_val_loss, val_metrics = self.validate()
            print(
                f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> 验证损失: {avg_val_loss:.4f}, PSNR: {val_metrics['psnr']:.2f}")

            self.scheduler.step(avg_val_loss)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                print(f"✓ 验证损失从 {best_val_loss:.4f} 降低到 {avg_val_loss:.4f}。正在保存模型...")
                best_val_loss = avg_val_loss
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model_self_supervised.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ 模型已保存至: {save_path}")

            # 保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_val_loss,
                }, checkpoint_path)

            # 更新日志文件
            self.update_log_file(epoch)

            # 绘制损失和指标曲线
            if (epoch + 1) % 5 == 0:
                self.plot_training_history()

        print("训练完成!")

    def validate(self):
        self.model.eval()
        total_loss = 0
        val_metrics = {'psnr': 0, 'rmse': 0, 'ssim': 0}

        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="[验证]"):
                if data is None or data[0] is None: continue
                left, right = data
                left, right = left.to(self.device), right.to(self.device)

                outputs = self.model(left, right)
                inputs = {"left_image": left, "right_image": right}
                loss_components = self.loss_fn(inputs, outputs)
                total_loss += loss_components["total_loss"].item()

                # 计算评估指标
                step_metrics = self.evaluator.evaluate_reconstruction(inputs, outputs, loss_components)
                for k in val_metrics:
                    val_metrics[k] += step_metrics[k]

        # 记录验证损失和指标
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        for k in self.loss_history['val']:
            self.loss_history['val'][k].append(loss_components[k].item() if k in loss_components else 0)

        for k in val_metrics:
            self.metric_history['val'][k].append(val_metrics[k] / len(self.val_loader))

        return avg_loss, {k: v / len(self.val_loader) for k, v in val_metrics.items()}

    def visualize(self, inputs, outputs, loss_components, step, phase="train"):
        left_img = inputs["left_image"][0].permute(1, 2, 0).cpu().numpy()
        right_img = inputs["right_image"][0].permute(1, 2, 0).cpu().numpy()
        pred_disp = outputs["disparity"][0, 0].cpu().detach().numpy()

        # 计算视差统计信息
        disp_min, disp_max = pred_disp.min(), pred_disp.max()
        disp_mean, disp_std = pred_disp.mean(), pred_disp.std()

        # 创建可视化图形
        fig = plt.figure(figsize=(20, 15))

        # 左图和右图
        plt.subplot(3, 4, 1)
        plt.imshow(left_img)
        plt.title("左图")
        plt.axis('off')

        plt.subplot(3, 4, 2)
        plt.imshow(right_img)
        plt.title("右图")
        plt.axis('off')

        # 预测视差图
        plt.subplot(3, 4, 3)
        disp_img = plt.imshow(pred_disp, cmap='viridis')
        plt.title(f"预测视差 (min:{disp_min:.2f}, max:{disp_max:.2f})")
        plt.axis('off')
        plt.colorbar(disp_img, fraction=0.046, pad=0.04)

        # 视差直方图
        plt.subplot(3, 4, 4)
        plt.hist(pred_disp.flatten(), bins=50)
        plt.title(f"视差分布 (μ:{disp_mean:.2f}, σ:{disp_std:.2f})")
        plt.xlabel("视差值")
        plt.ylabel("频次")

        # 重建图像和差异
        warped_right = loss_components["warped_right_image"][0].permute(1, 2, 0).cpu().detach().numpy()
        diff_right = np.clip(np.abs(left_img - warped_right), 0, 1)

        plt.subplot(3, 4, 5)
        plt.imshow(warped_right)
        plt.title("重建的左图 (从右图warp)")
        plt.axis('off')

        plt.subplot(3, 4, 6)
        diff_img = plt.imshow(diff_right, cmap='hot')
        plt.title("重建误差图")
        plt.axis('off')
        plt.colorbar(diff_img, fraction=0.046, pad=0.04)

        # 如果有双向重建，显示左图warp结果
        if "warped_left_image" in loss_components:
            warped_left = loss_components["warped_left_image"][0].permute(1, 2, 0).cpu().detach().numpy()
            diff_left = np.clip(np.abs(right_img - warped_left), 0, 1)

            plt.subplot(3, 4, 7)
            plt.imshow(warped_left)
            plt.title("重建的右图 (从左图warp)")
            plt.axis('off')

            plt.subplot(3, 4, 8)
            plt.imshow(diff_left, cmap='hot')
            plt.title("重建误差图 (右)")
            plt.axis('off')

        # 损失信息
        plt.subplot(3, 4, 9)
        text_content = [
            f"总损失: {loss_components['total_loss'].item():.4f}",
            f"光度损失: {loss_components['photometric_loss'].item():.4f}",
            f"平滑损失: {loss_components['smoothness_loss'].item():.4f}",
        ]
        if "consistency_loss" in loss_components:
            text_content.append(f"一致性损失: {loss_components['consistency_loss'].item():.4f}")

        plt.text(0.1, 0.9, "\n".join(text_content), transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.axis('off')
        plt.title("损失信息")

        # 评估指标
        metrics = self.evaluator.evaluate_reconstruction(inputs, outputs, loss_components)
        plt.subplot(3, 4, 10)
        metric_text = [
            f"PSNR: {metrics['psnr']:.2f} dB",
            f"RMSE: {metrics['rmse']:.4f}",
            f"SSIM: {metrics['ssim']:.4f}",
        ]
        plt.text(0.1, 0.9, "\n".join(metric_text), transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        plt.axis('off')
        plt.title("评估指标")

        plt.tight_layout()
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, f"{phase}_step_{step:06d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def plot_training_history(self):
        """绘制训练历史和指标曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.loss_history['train']['total'], label='训练损失')
        axes[0, 0].plot(self.loss_history['val']['total'], label='验证损失')
        axes[0, 0].set_title('总损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 损失组件
        axes[0, 1].plot(self.loss_history['train']['photo'], label='训练光度损失')
        axes[0, 1].plot(self.loss_history['val']['photo'], label='验证光度损失')
        axes[0, 1].plot(self.loss_history['train']['smooth'], label='训练平滑损失')
        axes[0, 1].plot(self.loss_history['val']['smooth'], label='验证平滑损失')
        axes[0, 1].set_title('损失组件')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('损失值')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # PSNR指标
        axes[1, 0].plot(self.metric_history['train']['psnr'], label='训练PSNR')
        axes[1, 0].plot(self.metric_history['val']['psnr'], label='验证PSNR')
        axes[1, 0].set_title('PSNR')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('dB')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # SSIM指标
        axes[1, 1].plot(self.metric_history['train']['ssim'], label='训练SSIM')
        axes[1, 1].plot(self.metric_history['val']['ssim'], label='验证SSIM')
        axes[1, 1].set_title('SSIM')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('相似度')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, "training_history.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def update_log_file(self, epoch):
        """更新日志文件"""
        log_data = {
            'config': self.cfg.__dict__,
            'epoch': epoch,
            'loss_history': self.loss_history,
            'metric_history': self.metric_history,
            'current_smoothness_weight': self.current_smoothness_weight,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'update_time': datetime.now().isoformat()
        }

        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)


# --- 7. 主执行模块 ---
if __name__ == '__main__':
    plt.switch_backend('Agg')
    config = Config()
    trainer = Trainer(config)
    trainer.train()