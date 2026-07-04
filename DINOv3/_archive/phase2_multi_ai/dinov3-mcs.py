# self_supervised_dinov3.py
# 实现了自监督的、端到端的波浪表面三维重建方案。
# (V6 最终性能版：引入多尺度监督，旨在突破性能瓶颈，达到最优效果)

import os
import sys
import glob
from dataclasses import dataclass

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
    MAX_DISPARITY: int = 126


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


# [核心修改] 损失函数现在支持多尺度监督
class SelfSupervisedLoss(nn.Module):
    def __init__(self, smoothness_weight=0.1, scales=4, scale_weights=None):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.ssim = SSIM()
        self.scales = scales
        # [修复] 调整权重列表以匹配3个尺度
        self.scale_weights = scale_weights if scale_weights is not None else [0.5, 0.7, 1.0]

    def forward(self, inputs, outputs):
        left_img = inputs["left_image"]
        right_img = inputs["right_image"]
        pred_disps = outputs["disparity"]  # 现在是一个列表

        total_loss = 0
        photometric_loss_sum = 0
        smoothness_loss_sum = 0

        for i, pred_disp in enumerate(pred_disps):
            # 将原图下采样到与当前尺度预测的视差图相同的尺寸
            h, w = pred_disp.shape[-2:]
            scaled_left = F.interpolate(left_img, (h, w), mode='bilinear', align_corners=False)
            scaled_right = F.interpolate(right_img, (h, w), mode='bilinear', align_corners=False)

            # 计算当前尺度的损失
            warped_right_image = self.inverse_warp(scaled_right, pred_disp)
            l1_loss = torch.abs(warped_right_image - scaled_left).mean()
            ssim_loss = self.ssim(warped_right_image, scaled_left).mean()
            photometric_loss = 0.85 * ssim_loss + 0.15 * l1_loss
            smoothness_loss = self.compute_smoothness_loss(pred_disp, scaled_left)

            # 将当前尺度的损失加权后计入总损失
            scale_loss = photometric_loss + self.smoothness_weight * smoothness_loss
            total_loss += self.scale_weights[i] * scale_loss

            photometric_loss_sum += photometric_loss
            smoothness_loss_sum += smoothness_loss

        return {
            "total_loss": total_loss,
            "photometric_loss": photometric_loss_sum / self.scales,
            "smoothness_loss": smoothness_loss_sum / self.scales,
            # 为了可视化，只返回最高分辨率的扭曲图像
            "warped_right_image": self.inverse_warp(right_img, pred_disps[-1])
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
        # 根据视差大小进行归一化，使平滑损失对不同尺度的视差范围不敏感
        disp = disp / (disp.mean() + 1e-8)
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()


# --- 3. PyTorch 数据集 ---
class WaveStereoSelfSupervisedDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
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
        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0
        return left_tensor, right_tensor


# --- 4. 深度学习模型 (核心架构重构) ---
def conv_block_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


class Hourglass3D(nn.Module):
    def __init__(self, in_channels):
        super(Hourglass3D, self).__init__()

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
        out_conv1 = self.conv1b(self.conv1a(x))
        out_conv2 = self.conv2b(self.conv2a(out_conv1))
        out_conv3 = self.conv3b(self.conv3a(out_conv2))

        up3 = F.interpolate(out_conv3, size=out_conv2.shape[2:], mode='trilinear', align_corners=False)
        deconv3_out = self.deconv3(up3)
        deconv2_in = F.relu(deconv3_out + self.redir2(out_conv2), inplace=True)

        up2 = F.interpolate(deconv2_in, size=out_conv1.shape[2:], mode='trilinear', align_corners=False)
        deconv2_out = self.deconv2(up2)
        deconv1_in = F.relu(deconv2_out + self.redir1(out_conv1), inplace=True)

        up1 = F.interpolate(deconv1_in, size=x.shape[2:], mode='trilinear', align_corners=False)
        deconv1_out = self.deconv1(up1)
        final_out = F.relu(deconv1_out + self.redir0(x), inplace=True)

        # [核心修改] 返回解码器每一层的输出，用于多尺度监督
        return [deconv2_in, deconv1_in, final_out]


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

        self.cost_aggregator = Hourglass3D(32)
        # [修复] 修正预测头的通道数并增加一个预测头
        self.disp_preds = nn.ModuleList([
            nn.Conv3d(32 * 4, 1, 3, padding=1),  # For deconv2_in (128 channels)
            nn.Conv3d(32 * 2, 1, 3, padding=1),  # For deconv1_in (64 channels)
            nn.Conv3d(32 * 1, 1, 3, padding=1),  # For final_out (32 channels)
        ])

        self.feature_conv = conv_block_3d(self.feature_dim, 32)
        print("模型构建完成 (多尺度3D沙漏网络架构)。")

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

        cost_volume = self.feature_conv(cost_volume)

        cost_aggregated_list = self.cost_aggregator(cost_volume)

        disparities = []
        for i, cost in enumerate(cost_aggregated_list):
            cost_pred = self.disp_preds[i](cost).squeeze(1)

            cost_softmax = F.softmax(-cost_pred, dim=1)
            disp_values = torch.arange(0, cost_softmax.shape[1], device=cost_softmax.device, dtype=torch.float32)
            disp_values = disp_values.view(1, -1, 1, 1)
            disparity_feat = torch.sum(cost_softmax * disp_values, 1, keepdim=True)

            disparity = F.interpolate(disparity_feat * self.patch_size, size=(h, w), mode='bilinear',
                                      align_corners=False)
            disparities.append(disparity)

        return {"disparity": disparities}


# --- 5. 训练器 ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    return torch.utils.data.dataloader.default_collate(batch)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        if cfg.VISUALIZE_TRAINING: os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)
        if not torch.cuda.is_available(): sys.exit("【致命错误】: 未检测到GPU。")
        self.device = torch.device("cuda")
        print(f"✓ 成功检测到GPU，将使用设备: {self.device}")

        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.loss_fn = SelfSupervisedLoss(smoothness_weight=cfg.SMOOTHNESS_LOSS_WEIGHT,
                                          scales=len(self.model.disp_preds))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)

        train_dataset = WaveStereoSelfSupervisedDataset(cfg, is_validation=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                                       num_workers=4, pin_memory=True)
        val_dataset = WaveStereoSelfSupervisedDataset(cfg, is_validation=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=4, pin_memory=True)
        self.step = 0
        self._setup_visualization_font()

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
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [训练]")
            for data in pbar:
                if data is None or data[0] is None: continue
                left, right = data
                left, right = left.to(self.device), right.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(left, right)
                inputs = {"left_image": left, "right_image": right}
                loss_components = self.loss_fn(inputs, outputs)
                loss = loss_components["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRADIENT_CLIP_VAL)
                self.optimizer.step()
                pbar.set_postfix({'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']})
                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(inputs, outputs, loss_components, self.step)
                self.step += 1
            avg_val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> 验证损失: {avg_val_loss:.4f}")
            self.scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                print(f"✓ 验证损失从 {best_val_loss:.4f} 降低到 {avg_val_loss:.4f}。正在保存模型...")
                best_val_loss = avg_val_loss
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model_self_supervised.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ 模型已保存至: {save_path}")

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="[验证]"):
                if data is None or data[0] is None: continue
                left, right = data
                left, right = left.to(self.device), right.to(self.device)
                outputs = self.model(left, right)
                inputs = {"left_image": left, "right_image": right}
                loss_components = self.loss_fn(inputs, outputs)
                total_loss += loss_components["total_loss"].item()
        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

    def visualize(self, inputs, outputs, loss_components, step):
        left_img = inputs["left_image"][0].permute(1, 2, 0).cpu().numpy()
        right_img = inputs["right_image"][0].permute(1, 2, 0).cpu().numpy()
        # 可视化最高分辨率的视差图
        pred_disp = outputs["disparity"][-1][0, 0].cpu().detach().numpy()
        warped_right = loss_components["warped_right_image"][0].permute(1, 2, 0).cpu().detach().numpy()
        vmax = np.percentile(pred_disp, 95)
        pred_disp_color = np.clip(pred_disp, 0, vmax)
        pred_disp_color = cv2.normalize(pred_disp_color, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        pred_disp_color = cv2.applyColorMap(pred_disp_color, cv2.COLORMAP_JET)
        pred_disp_color = cv2.cvtColor(pred_disp_color, cv2.COLOR_BGR2RGB)
        diff = np.clip(np.abs(left_img - warped_right), 0, 1)
        fig, axes = plt.subplots(5, 1, figsize=(10, 25))
        axes[0].imshow(left_img);
        axes[0].set_title("左图 (目标)")
        axes[1].imshow(right_img);
        axes[1].set_title("右图 (源)")
        axes[2].imshow(pred_disp_color);
        axes[2].set_title("预测的视差图")
        axes[3].imshow(warped_right);
        axes[3].set_title("变换后的右图 (重建的伪左图)")
        axes[4].imshow(diff);
        axes[4].set_title("差异图 (左图 vs 伪左图)")
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, f"step_{step:06d}.png")
        plt.savefig(save_path)
        plt.close(fig)


# --- 6. 主执行模块 ---
if __name__ == '__main__':
    plt.switch_backend('Agg')
    config = Config()
    trainer = Trainer(config)
    trainer.train()
