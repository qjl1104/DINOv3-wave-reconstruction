# self_supervised_dinov3.py
# 实现了自监督的、端到端的波浪表面三维重建方案。
# (最终修复版：内置GPU兼容性诊断与最终修复指令)

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

# --- 使用AutoModel以智能加载DINOv3 ---
try:
    from transformers import AutoModel
except ImportError:
    print("=" * 80)
    print("【致命错误】: 无法从 'transformers' 库中导入 'AutoModel'。")
    print("这表明您的Conda环境存在严重问题，或者库没有正确安装。")
    print("\n请严格按照以下【环境修复指令】操作来解决问题：")
    print("\n--- 环境修复指令 ---")
    print("1. 打开 Anaconda Prompt 终端。")
    print("2. 激活您的环境: \n   conda activate dino")
    print("3. 如果transformers库已安装但损坏，请先卸载: \n   pip uninstall transformers")
    print("4. 从源代码重新安装最新的库: \n   pip install git+https://github.com/huggingface/transformers")
    print("=" * 80)
    sys.exit(1)  # 遇到严重错误，直接退出


# --- 1. 配置中心 ---
@dataclass
class Config:
    """项目配置参数"""
    # --- 文件路径 (请根据您的实际环境进行配置) ---
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
    VISUALIZE_INTERVAL: int = 100  # 每100步保存一次可视化结果

    # --- 数据处理参数 (必须是14的倍数以匹配ViT的patch size) ---
    IMAGE_HEIGHT: int = 252
    IMAGE_WIDTH: int = 504

    # --- 模型与训练参数 ---
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1


# --- 2. 自监督损失函数 ---
class SelfSupervisedLoss(nn.Module):
    """
    计算自监督立体匹配的损失，主要包含光度损失和平滑度损失。
    该损失函数在特征空间中计算光度一致性，而不是像素空间。
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, outputs):
        left_img_features = inputs["left_features"]
        right_img_features = inputs["right_features"]
        pred_disp = outputs["disparity"]

        # 步骤1: 根据预测的视差，将右图特征“扭曲”到左图的视角
        warped_right_features = self.inverse_warp(right_img_features, pred_disp)

        # 步骤2: 计算光度损失 (Photometric Loss)，但在DINOv3的特征空间中进行
        photometric_loss = self.compute_feature_metric_loss(warped_right_features, left_img_features)

        # 步骤3: 计算平滑度损失 (Smoothness Loss)，鼓励视差图平滑，但在图像边缘处允许突变
        # 使用原始左图作为边缘的指导
        smoothness_loss = self.compute_smoothness_loss(pred_disp, inputs["left_image"])

        # 步骤4: 组合损失，平滑度损失的权重通常较小
        total_loss = photometric_loss + 0.1 * smoothness_loss

        loss_components = {
            "total_loss": total_loss,
            "photometric_loss": photometric_loss,
            "smoothness_loss": smoothness_loss,
            # 同时生成用于可视化的扭曲图像
            "warped_right_image": self.inverse_warp(inputs["right_image"], pred_disp)
        }
        return loss_components

    def inverse_warp(self, features, disp):
        """根据视差图对特征图或图像进行反向扭曲（warping）"""
        B, C, H, W = features.shape
        # 创建一个像素坐标网格
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=0).float().to(features.device)
        pixel_coords = pixel_coords.repeat(B, 1, 1, 1)

        disp = disp.squeeze(1)  # 形状从 [B, 1, H, W] 变为 [B, H, W]
        # 新的x坐标 = 原始x坐标 - 视差
        transformed_x = pixel_coords[:, 0, :, :] - disp

        # 组合成新的采样坐标网格
        grid = torch.stack([transformed_x, pixel_coords[:, 1, :, :]], dim=-1)

        # 将坐标归一化到 F.grid_sample 所需的 [-1, 1] 范围
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

        # 使用双线性插值进行采样
        warped_features = F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_features

    def compute_feature_metric_loss(self, pred, target):
        """在DINO特征空间中计算L1损失，比像素级L1损失更鲁棒"""
        return torch.abs(pred - target).mean()

    def compute_smoothness_loss(self, disp, img):
        """计算边缘感知的平滑度损失"""
        # 计算视差图在x和y方向的梯度
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        # 计算原图在x和y方向的梯度，作为边缘的权重
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        # 使用图像梯度对视差梯度进行加权，图像梯度大的地方（边缘），平滑约束就弱
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


# --- 3. PyTorch 数据集 ---
class WaveStereoSelfSupervisedDataset(Dataset):
    def __init__(self, cfg: Config, is_validation=False):
        self.cfg = cfg
        self.left_images = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))

        # 数据集划分
        num_frames = len(self.left_images)
        indices = np.arange(num_frames)
        np.random.seed(42)  # 固定随机种子以保证每次划分一致
        np.random.shuffle(indices)
        split_idx = int(num_frames * (1 - cfg.VALIDATION_SPLIT))
        self.indices = indices[split_idx:] if is_validation else indices[:split_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        left_img_path = self.left_images[frame_idx]

        # 根据左图路径推断右图路径
        frame_basename = os.path.basename(left_img_path)
        right_frame_basename = frame_basename.replace('lresult', 'rresult', 1)
        right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_frame_basename)

        try:
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)
            if left_img is None or right_img is None:
                # 如果图像读取失败，打印警告并返回None
                print(f"警告: 无法读取图像对: {left_img_path} 或 {right_img_path}")
                return None
        except Exception as e:
            print(f"警告: 读取图像时发生错误 {e}")
            return None

        # 图像预处理
        target_h, target_w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
        left_img = cv2.resize(left_img, (target_w, target_h))
        right_img = cv2.resize(right_img, (target_w, target_h))

        # 转换为Tensor并归一化
        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0

        return left_tensor, right_tensor


# --- 4. 深度学习模型 ---
class DINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # 加载DINOv3作为特征提取器
        self.dino = self._load_dino_model()
        if self.dino is None:
            raise RuntimeError("无法加载DINOv3模型，请检查网络连接或本地模型路径。")

        # 冻结DINOv3的参数，我们只用它来提取特征，不训练它
        for param in self.dino.parameters():
            param.requires_grad = False

        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 4)

        print(
            f"DINOv3配置: 特征维度={self.feature_dim}, Patch Size={self.patch_size}, 注册令牌={self.num_register_tokens}")

        # 构建一个轻量级的解码器来从特征预测视差
        self.decoder = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 使用转置卷积进行上采样
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )
        print("模型构建完成。")

    def _load_dino_model(self):
        # 优先从本地加载，避免重复下载
        if os.path.exists(self.cfg.DINO_LOCAL_PATH):
            try:
                print(f"尝试从本地路径加载模型: {self.cfg.DINO_LOCAL_PATH}")
                model = AutoModel.from_pretrained(self.cfg.DINO_LOCAL_PATH, local_files_only=True)
                print("✓ 成功从本地加载DINOv3模型")
                return model
            except Exception as e:
                print(f"[!] 从本地加载模型失败: {e}。将尝试从网络下载。")

        # 如果本地加载失败，则从Hugging Face Hub下载并保存到本地
        try:
            print(f"尝试从Hugging Face Hub加载模型: {self.cfg.DINO_ONLINE_MODEL}")
            model = AutoModel.from_pretrained(self.cfg.DINO_ONLINE_MODEL)
            print("✓ 成功从Hugging Face Hub加载DINOv3模型")
            model.save_pretrained(self.cfg.DINO_LOCAL_PATH)
            print(f"模型已自动保存到本地: {self.cfg.DINO_LOCAL_PATH}")
            return model
        except Exception as e:
            print(f"[!] 从Hugging Face Hub加载模型失败: {e}")
            return None

    def get_features(self, image):
        b, c, h, w = image.shape
        with torch.no_grad():  # 确保不计算梯度
            outputs = self.dino(image)
            features = outputs.last_hidden_state

        # 关键步骤：跳过[CLS]和注册令牌，只提取图像块的特征
        start_index = 1 + self.num_register_tokens
        patch_tokens = features[:, start_index:, :]

        # 将一维的patch序列重新整形为二维的特征图
        feature_h, feature_w = h // self.patch_size, w // self.patch_size
        features_2d = patch_tokens.permute(0, 2, 1).reshape(b, self.feature_dim, feature_h, feature_w)

        # 上采样特征图到与输入图像相同的分辨率，以便后续处理
        features_2d = F.interpolate(features_2d, size=(h, w), mode='bilinear', align_corners=False)
        return features_2d

    def forward(self, left_image, right_image):
        h, w = left_image.shape[-2:]

        # 提取左图的DINO特征
        left_features = self.get_features(left_image)

        # 仅用左图特征通过解码器预测视差
        normalized_disparity = self.decoder(left_features)

        # 将归一化的视差图上采样到原始图像尺寸
        disparity = F.interpolate(normalized_disparity, size=(h, w), mode='bilinear', align_corners=False)

        # 将视差从[0, 1]范围缩放到一个合理的像素范围，例如图像宽度的20%
        max_disparity = w * 0.2
        disparity = disparity * max_disparity

        return {"disparity": disparity}


# --- 5. 训练器 ---
def collate_fn(batch):
    """
    自定义的数据整理函数，用于过滤掉数据集中返回None的无效样本。
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        if cfg.VISUALIZE_TRAINING:
            os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)
            print(f"训练过程可视化结果将保存在: {cfg.VISUALIZATION_DIR}")

        # --- 强制GPU检查与环境修复指南 ---
        if not torch.cuda.is_available():
            print("=" * 80)
            print("【致命错误】: PyTorch无法找到或使用您的NVIDIA GPU！")
            print("脚本已终止。在CPU上运行此模型速度极慢，没有实际意义。")
            print("\n【问题诊断】: 这个问题几乎总是由Conda环境配置错误引起的。")
            print("您当前的PyTorch版本与您的显卡驱动或CUDA工具包不兼容。")
            print("\n请严格按照以下【最终解决方案】操作，这将彻底解决问题：")
            print("\n--- 最终解决方案：使用Conda净化并重装 ---")
            print("1. 打开 Anaconda Prompt 终端。")
            print("2. 停用并彻底移除旧环境: \n   conda deactivate\n   conda env remove -n dino")
            print("3. 【关键】清除所有Conda缓存，确保全新安装: \n   conda clean --all -y")
            print("4. 重新创建纯净环境并安装Python: \n   conda create -n dino python=3.10 -y\n   conda activate dino")
            print(
                "5. 【关键】使用官方渠道安装与CUDA 12.1兼容的PyTorch: \n   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
            print("6. 安装其余必需的库: \n   pip install opencv-python scipy tqdm matplotlib scikit-learn")
            print(
                "7. 【关键】从源代码安装最新的transformers库: \n   pip install git+https://github.com/huggingface/transformers")
            print("=" * 80)
            sys.exit(1)  # 诊断后直接退出，强制用户修复环境

        self.device = torch.device("cuda")
        print(f"✓ 成功检测到GPU，将使用设备: {self.device}")

        self.model = DINOv3StereoModel(cfg).to(self.device)
        # 只将需要训练的参数（解码器部分）传递给优化器
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=cfg.LEARNING_RATE)
        self.loss_fn = SelfSupervisedLoss()

        train_dataset = WaveStereoSelfSupervisedDataset(cfg, is_validation=False)
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                                       num_workers=4, pin_memory=True)

        val_dataset = WaveStereoSelfSupervisedDataset(cfg, is_validation=True)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=4, pin_memory=True)

        self.step = 0
        self._setup_visualization_font()

    def _setup_visualization_font(self):
        """智能查找并设置用于可视化的中文字体，避免乱码。"""
        # 常见中文字体列表
        font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'sans-serif']
        for font_name in font_names:
            try:
                # 检查字体是否存在
                if any(font.name == font_name for font in fm.fontManager.ttflist):
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    print(f"✓ 可视化已配置中文字体: {font_name}")
                    return
            except Exception:
                continue
        print("警告: 未找到指定的中文字体 (SimHei, Microsoft YaHei等)，可视化结果中的中文可能显示为方块。")

    def train(self):
        print("--- 开始自监督训练 ---")
        best_val_loss = float('inf')
        for epoch in range(self.cfg.NUM_EPOCHS):
            self.model.train()  # 设置为训练模式
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} [训练]")
            for data in pbar:
                if data is None or data[0] is None: continue  # 跳过无效的批次
                left, right = data
                left, right = left.to(self.device), right.to(self.device)

                self.optimizer.zero_grad()

                # 在自监督模式下，我们需要左右图的特征来计算损失
                left_features = self.model.get_features(left)
                right_features = self.model.get_features(right)

                # 模型前向传播，只使用左图预测视差
                outputs = self.model(left, right)

                # 准备计算损失所需的所有输入
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

            # 每个epoch结束后进行验证
            avg_val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS} -> 验证损失: {avg_val_loss:.4f}")

            # 如果验证损失降低，则保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model_self_supervised.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ 验证损失提升，模型已保存至: {save_path}")

    def validate(self):
        self.model.eval()  # 设置为评估模式
        total_loss = 0
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="[验证]"):
                if data is None or data[0] is None: continue
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
        # 从批次中取出第一个样本进行可视化
        left_img = inputs["left_image"][0].permute(1, 2, 0).cpu().numpy()
        right_img = inputs["right_image"][0].permute(1, 2, 0).cpu().numpy()
        pred_disp = outputs["disparity"][0, 0].cpu().detach().numpy()
        warped_right = loss_components["warped_right_image"][0].permute(1, 2, 0).cpu().detach().numpy()

        # 将单通道的视差图转换为彩色的伪彩色图以便观察
        pred_disp_color = cv2.normalize(pred_disp, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        pred_disp_color = cv2.applyColorMap(pred_disp_color, cv2.COLORMAP_JET)
        pred_disp_color = cv2.cvtColor(pred_disp_color, cv2.COLOR_BGR2RGB)  # Matplotlib使用RGB

        # 计算差异图
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
    # 在Windows上使用多进程(num_workers>0)时，必须将代码放在这个保护块内
    # 设置matplotlib后端为'Agg'，这样可以在没有图形界面的服务器上运行并保存图像
    plt.switch_backend('Agg')

    config = Config()
    trainer = Trainer(config)
    trainer.train()
