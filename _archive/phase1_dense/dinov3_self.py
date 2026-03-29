# self_supervised_dinov3.py
# 实现了自监督的、端到端的波浪表面三维重建方案。
# (版本已根据用户代码升级为DINOv3)

import os
import glob
import pickle
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

# --- 使用AutoModel以智能加载DINOv3 ---
try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80)
    print("错误: 无法从 'transformers' 中导入 'AutoModel'。")
    print("这表明您的Conda环境依然存在问题。")
    print("\n请严格按照以下【环境修复指令】操作：")
    print("\n--- 环境修复指令 ---")
    print("1. 打开Anaconda Prompt终端。")
    print("2. 激活您的环境: \n   conda activate dino")
    print("3. 从源代码安装最新的库: \n   pip install git+https://github.com/huggingface/transformers")
    print("=" * 80)
    exit()


# --- 1. 配置中心 ---
@dataclass
class Config:
    """项目配置参数"""
    # --- 文件路径 (已根据您的环境配置) ---
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
    IMAGE_HEIGHT: int = 252  # 必须是14的倍数
    IMAGE_WIDTH: int = 504  # 必须是14的倍数

    # --- 模型与训练参数 ---
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1


# --- 2. 自监督损失函数 ---
class SelfSupervisedLoss(nn.Module):
    """
    计算自监督立体匹配的损失，主要包含光度损失和平滑度损失。
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, outputs):
        left_img_features = inputs["left_features"]
        right_img_features = inputs["right_features"]
        pred_disp = outputs["disparity"]

        # 步骤1: 生成变换后的右图特征 (warped_right_features)
        warped_right_features = self.inverse_warp(right_img_features, pred_disp)

        # 步骤2: 计算光度损失 (Photometric Loss)，但在特征空间中进行
        photometric_loss = self.compute_feature_metric_loss(warped_right_features, left_img_features)

        # 步骤3: 计算平滑度损失 (Smoothness Loss)
        # 使用原始左图作为边缘指导
        smoothness_loss = self.compute_smoothness_loss(pred_disp, inputs["left_image"])

        # 步骤4: 组合损失
        total_loss = photometric_loss + 0.1 * smoothness_loss

        loss_components = {
            "total_loss": total_loss,
            "photometric_loss": photometric_loss,
            "smoothness_loss": smoothness_loss,
            "warped_right_image": self.inverse_warp(inputs["right_image"], pred_disp)  # 用于可视化
        }
        return loss_components

    def inverse_warp(self, features, disp):
        """根据视差图对特征图或图像进行变换"""
        B, C, H, W = features.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=0).float().to(features.device)
        pixel_coords = pixel_coords.repeat(B, 1, 1, 1)

        disp = disp.squeeze(1)  # [B, H, W]
        transformed_x = pixel_coords[:, 0, :, :] - disp

        # 组合成新的坐标网格
        grid = torch.stack([transformed_x, pixel_coords[:, 1, :, :]], dim=-1)

        # 归一化坐标到 [-1, 1]
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

        warped_features = F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_features

    def compute_feature_metric_loss(self, pred, target):
        """在DINO特征空间中计算L1损失"""
        return torch.abs(pred - target).mean()

    def compute_smoothness_loss(self, disp, img):
        """计算边缘感知的平滑度损失"""
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


# --- 3. PyTorch Dataset ---
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
            if left_img is None or right_img is None: raise FileNotFoundError
        except (FileNotFoundError, TypeError, IOError):
            return None

        target_h, target_w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
        left_img = cv2.resize(left_img, (target_w, target_h))
        right_img = cv2.resize(right_img, (target_w, target_h))

        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0

        return left_tensor, right_tensor


# --- 4. 深度学习模型 ---
class DINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.dino = self._load_dino_model()
        if self.dino is None:
            raise RuntimeError("无法加载DINOv3模型")

        for param in self.dino.parameters():
            param.requires_grad = False

        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)

        print(
            f"DINOv3配置: 特征维度={self.feature_dim}, Patch Size={self.patch_size}, 注册令牌={self.num_register_tokens}")

        self.decoder = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        print("模型构建完成。")

    def _load_dino_model(self):
        # 优先从本地加载
        if os.path.exists(self.cfg.DINO_LOCAL_PATH):
            try:
                print(f"尝试从本地路径加载模型: {self.cfg.DINO_LOCAL_PATH}")
                model = AutoModel.from_pretrained(self.cfg.DINO_LOCAL_PATH, local_files_only=True)
                print("✓ 成功从本地加载DINOv3模型")
                return model
            except Exception as e:
                print(f"[!] 从本地加载模型失败: {e}。将尝试从网络下载。")

        # 如果本地加载失败，则从Hugging Face Hub下载并保存
        try:
            print(f"尝试从Hugging Face Hub加载模型: {self.cfg.DINO_ONLINE_MODEL}")
            model = AutoModel.from_pretrained(self.cfg.DINO_ONLINE_MODEL)
            print("✓ 成功从Hugging Face Hub加载DINOv3模型")
            model.save_pretrained(self.cfg.DINO_LOCAL_PATH)
            print(f"模型已保存到本地: {self.cfg.DINO_LOCAL_PATH}")
            return model
        except Exception as e:
            print(f"[!] 从Hugging Face Hub加载模型失败: {e}")
            return None

    def get_features(self, image):
        b, c, h, w = image.shape
        with torch.no_grad():
            outputs = self.dino(image)
            features = outputs.last_hidden_state

        # 关键：跳过CLS和注册令牌
        start_index = 1 + self.num_register_tokens
        patch_tokens = features[:, start_index:, :]

        feature_h, feature_w = h // self.patch_size, w // self.patch_size
        features_2d = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, feature_h, feature_w)

        # 上采样到与输入图像相同的分辨率
        features_2d = F.interpolate(features_2d, size=(h, w), mode='bilinear', align_corners=True)
        return features_2d

    def forward(self, left_image, right_image):
        h, w = left_image.shape[-2:]

        # 提取左右图的DINO特征
        left_features = self.get_features(left_image)

        # 仅用左图特征预测视差
        normalized_disparity = self.decoder(left_features)

        # 将视差放大回原始尺寸
        disparity = F.interpolate(normalized_disparity, size=(h, w), mode='bilinear', align_corners=True)

        max_disparity = w * 0.2
        disparity = disparity * max_disparity

        return {"disparity": disparity}


# --- 5. 训练器 ---
def collate_fn(batch):
    """
    过滤掉数据集中返回None的无效样本。
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        if cfg.VISUALIZE_TRAINING:
            os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)
            print(f"训练过程可视化结果将保存在: {cfg.VISUALIZATION_DIR}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.model = DINOv3StereoModel(cfg).to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.loss_fn = SelfSupervisedLoss()

        train_dataset = WaveStereoSelfSupervisedDataset(cfg, is_validation=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                                       num_workers=0)

        val_dataset = WaveStereoSelfSupervisedDataset(cfg, is_validation=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=0)

        self.step = 0

    def train(self):
        print("--- 开始自监督训练 ---")
        best_val_loss = float('inf')
        for epoch in range(self.cfg.NUM_EPOCHS):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [训练]")
            for data in pbar:
                if not data: continue
                left, right = data
                left, right = left.to(self.device), right.to(self.device)

                self.optimizer.zero_grad()

                left_features = self.model.get_features(left)
                right_features = self.model.get_features(right)

                outputs = self.model(left, right)

                inputs = {"left_image": left, "right_image": right, "left_features": left_features,
                          "right_features": right_features}
                loss_components = self.loss_fn(inputs, outputs)

                loss = loss_components["total_loss"]
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix({'loss': loss.item()})

                if self.cfg.VISUALIZE_TRAINING and self.step % self.cfg.VISUALIZE_INTERVAL == 0:
                    self.visualize(inputs, outputs, loss_components, self.step)

                self.step += 1

            avg_val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> 验证损失: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model_self_supervised.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"验证损失提升，模型已保存至: {save_path}")

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="[验证]"):
                if not data: continue
                left, right = data
                left, right = left.to(self.device), right.to(self.device)

                left_features = self.model.get_features(left)
                right_features = self.model.get_features(right)

                outputs = self.model(left, right)
                inputs = {"left_image": left, "right_image": right, "left_features": left_features,
                          "right_features": right_features}
                loss_components = self.loss_fn(inputs, outputs)
                total_loss += loss_components["total_loss"].item()
        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

    def visualize(self, inputs, outputs, loss_components, step):
        """生成并保存在线可视化的结果"""
        left_img = inputs["left_image"][0].permute(1, 2, 0).cpu().numpy()
        right_img = inputs["right_image"][0].permute(1, 2, 0).cpu().numpy()
        pred_disp = outputs["disparity"][0, 0].cpu().detach().numpy()
        warped_right = loss_components["warped_right_image"][0].permute(1, 2, 0).cpu().detach().numpy()

        pred_disp_color = cv2.normalize(pred_disp, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        pred_disp_color = cv2.applyColorMap(pred_disp_color, cv2.COLORMAP_JET)

        diff = np.clip(np.abs(left_img - warped_right), 0, 1)

        fig, axes = plt.subplots(5, 1, figsize=(10, 25))
        axes[0].imshow(left_img);
        axes[0].set_title("左图 (目标)")
        axes[1].imshow(right_img);
        axes[1].set_title("右图 (源)")
        axes[2].imshow(pred_disp_color);
        axes[2].set_title("预测的视差图")
        axes[3].imshow(warped_right);
        axes[3].set_title("变换后的右图 (伪左图)")
        axes[4].imshow(diff);
        axes[4].set_title("差异图 (左图 vs 伪左图)")

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, f"step_{step:06d}.png")
        plt.savefig(save_path)
        plt.close(fig)


# --- 6. 主执行模块 ---
if __name__ == '__main__':
    # 注意：PyTorch的多进程在Windows上需要这个 if __name__ == '__main__': 保护块
    plt.switch_backend('Agg')
    config = Config()
    trainer = Trainer(config)
    trainer.train()
