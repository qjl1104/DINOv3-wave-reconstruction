# dinov3_path_final.py
# 实现了从含泡沫图像直接重建无泡沫波浪表面的端到端深度学习方案
# (最终修正版：自动化真值生成与训练流程)

import os
import sys
import glob
import pickle
from dataclasses import dataclass

# --- 环境诊断代码 ---
print("=" * 80)
print("--- 正在执行环境诊断 ---")
print(f"[*] 当前使用的 Python 解释器: {sys.executable}")
try:
    import transformers

    print(f"[*] 'transformers' 库已找到。")
    print(f"    - 版本: {transformers.__version__}")
    print(f"    - 安装路径: {transformers.__path__}")
except ImportError:
    print("[!] 错误: 未找到 'transformers' 库。这表明它没有在当前解释器环境中正确安装。")
    pass
print("=" * 80)
# --- 诊断代码结束 ---


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import curve_fit, linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 导入检查 (已修正) ---
try:
    # 关键修正：增加 AutoModel，它会自动选择正确的模型架构
    from transformers import Dinov2Model, Dinov2Config, AutoConfig, AutoModel
except ImportError:
    print("[!] 错误: 无法导入所需模块。请确保 transformers 库已正确安装。")
    exit()


# --- 1. 配置中心 (增加了模型文件检查) ---
@dataclass
class Config:
    """项目配置参数"""
    LEFT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/lresult/"
    RIGHT_IMAGE_DIR: str = "D:/Research/wave_reconstruction_project/data/rresult/"
    CALIBRATION_FILE: str = "D:/Research/wave_reconstruction_project/camera_calibration/params/stereo_calib_params_from_matlab_full.npz"
    GROUND_TRUTH_DIR: str = "D:/Research/wave_reconstruction_project/data/ground_truth_disparity/"
    CHECKPOINT_DIR: str = "./checkpoints/"
    VISUALIZATION_DIR: str = "D:/Research/wave_reconstruction_project/data/visualization/"
    VISUALIZE_GT_GENERATION: bool = True
    VISUALIZE_FRAME_INTERVAL: int = 50
    IMAGE_HEIGHT: int = 320
    IMAGE_WIDTH: int = 512
    DINO_MODEL_NAME: str = "./dinov3-base-model/"
    DINO_ONLINE_MODEL: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50
    VALIDATION_SPLIT: float = 0.1

    def check_model_files(self):
        """检查模型文件是否存在"""
        required_files = ["config.json", "pytorch_model.bin"]
        model_path = self.DINO_MODEL_NAME

        if not os.path.exists(model_path):
            print(f"[!] 模型路径不存在: {model_path}")
            return False

        existing_files = os.listdir(model_path)
        missing_files = [f for f in required_files if f not in existing_files]

        if missing_files:
            print(f"[!] 模型文件缺失: {missing_files}")
            return False

        print(f"[✓] 模型文件检查通过: {model_path}")
        return True


# --- 2. 真值生成器 (已补全功能) ---
class GroundTruthGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.GROUND_TRUTH_DIR, exist_ok=True)
        if cfg.VISUALIZE_GT_GENERATION:
            os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)
        calib = np.load(cfg.CALIBRATION_FILE)
        self.P1, self.P2 = calib['P1'], calib['P2']
        self.focal_length = self.P1[0, 0]
        self.baseline = -self.P2[0, 3] / self.P1[0, 0]

    @staticmethod
    def detect_markers(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.array([])
        _, binary_img = cv2.threshold(img, 68, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker_centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 8 < area < 500 and len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    hull = cv2.convexHull(contour)
                    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                    (w, h) = ellipse[1]
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100
                    if solidity > 0.85 and aspect_ratio < 8.0:
                        marker_centers.append(ellipse[0])
                except cv2.error:
                    continue
        return np.array(marker_centers, dtype=np.float32)

    def match_and_triangulate(self, left_markers: np.ndarray, right_markers: np.ndarray) -> np.ndarray:
        if left_markers.size == 0 or right_markers.size == 0: return np.array([])
        cost_matrix = np.full((len(left_markers), len(right_markers)), np.inf)
        for i, p_l in enumerate(left_markers):
            for j, p_r in enumerate(right_markers):
                if abs(p_l[1] - p_r[1]) < 10 and p_l[0] > p_r[0]:
                    cost_matrix[i, j] = np.linalg.norm(p_l - p_r)

        # --- 【关键修正】 ---
        # 如果没有任何一对标记点满足约束，成本矩阵将全为inf，导致算法失败。
        # 在调用前检查这种情况，如果为真，则返回空数组，跳过此帧。
        if np.all(np.isinf(cost_matrix)):
            return np.array([])
        # --- 【修正结束】 ---

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_points_l, matched_points_r = [], []
        for r, c in zip(row_ind, col_ind):
            if np.isfinite(cost_matrix[r, c]):
                matched_points_l.append(left_markers[r])
                matched_points_r.append(right_markers[c])
        if not matched_points_l: return np.array([])
        points_l_np, points_r_np = np.array(matched_points_l).T, np.array(matched_points_r).T
        points_4d_hom = cv2.triangulatePoints(self.P1, self.P2, points_l_np, points_r_np)
        return (points_4d_hom[:3] / (points_4d_hom[3] + 1e-8)).T

    @staticmethod
    def wave_surface_func(xy_data, a, fx, px, fy, py, o):
        x, y = xy_data[0, :], xy_data[1, :]
        return a * np.cos(fx * x + px) * np.cos(fy * y + py) + o

    def fit_and_generate_dense_map(self, points_3d: np.ndarray, orig_shape: tuple) -> tuple:
        if len(points_3d) < 10: return None, None
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        try:
            p0 = [np.std(z), 0.1, 0, 0.1, 0, np.mean(z)]
            params, _ = curve_fit(self.wave_surface_func, np.vstack((x, y)), z, p0=p0, maxfev=5000)
        except RuntimeError:
            return None, None
        h, w = orig_shape
        yh, xh = np.mgrid[0:h, 0:w]
        K_inv = np.linalg.inv(self.P1[:, :3])
        px_coords = np.stack([xh.flatten(), yh.flatten(), np.ones_like(xh.flatten())], axis=1)
        cam_coords = (K_inv @ px_coords.T).T
        world_x, world_y = cam_coords[:, 0] * np.mean(z), cam_coords[:, 1] * np.mean(z)
        dense_z = self.wave_surface_func(np.vstack((world_x, world_y)), *params).reshape(h, w)
        dense_z[dense_z <= 0] = 1e-6
        disparity_map = (self.focal_length * self.baseline) / dense_z
        return disparity_map.astype(np.float32), params

    def process_all_frames(self):
        print("--- 开始生成真值视差图 ---")
        left_image_paths = sorted(glob.glob(os.path.join(self.cfg.LEFT_IMAGE_DIR, "*.*")))

        for i, l_path in enumerate(tqdm(left_image_paths, desc="处理帧")):
            basename = os.path.basename(l_path)
            r_basename = basename.replace('lresult', 'rresult')
            r_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, r_basename)

            gt_filename = f"gt_disparity_{basename.split('.')[0]}.pkl"
            gt_filepath = os.path.join(self.cfg.GROUND_TRUTH_DIR, gt_filename)

            if os.path.exists(gt_filepath):
                continue

            if not os.path.exists(r_path):
                continue

            left_markers = self.detect_markers(l_path)
            right_markers = self.detect_markers(r_path)

            if len(left_markers) < 10 or len(right_markers) < 10:
                continue

            points_3d = self.match_and_triangulate(left_markers, right_markers)
            if len(points_3d) < 10:
                continue

            left_img_for_shape = cv2.imread(l_path)
            disp_map, fit_params = self.fit_and_generate_dense_map(points_3d, left_img_for_shape.shape[:2])

            if disp_map is not None:
                with open(gt_filepath, 'wb') as f:
                    pickle.dump(disp_map, f)

                if self.cfg.VISUALIZE_GT_GENERATION and i % self.cfg.VISUALIZE_FRAME_INTERVAL == 0:
                    self.visualize_results(l_path, r_path, left_markers, right_markers, points_3d, disp_map, fit_params,
                                           basename)
        print("--- 真值生成完毕 ---")

    def visualize_results(self, l_path, r_path, l_markers, r_markers, points_3d, disp_map, fit_params, basename):
        l_img = cv2.imread(l_path)
        r_img = cv2.imread(r_path)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"真值生成可视化: {basename}", fontsize=16)

        ax = axes[0, 0]
        l_img_disp = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
        for x, y in l_markers:
            cv2.circle(l_img_disp, (int(x), int(y)), 5, (0, 255, 0), -1)
        ax.imshow(l_img_disp)
        ax.set_title("左图标注点")
        ax.axis('off')

        ax = axes[0, 1]
        r_img_disp = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        for x, y in r_markers:
            cv2.circle(r_img_disp, (int(x), int(y)), 5, (0, 255, 0), -1)
        ax.imshow(r_img_disp)
        ax.set_title("右图标注点")
        ax.axis('off')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap='viridis')
        ax.set_title("三维重建点云")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax = axes[1, 1]
        im = ax.imshow(disp_map, cmap='jet')
        ax.set_title("生成的视差图 (真值)")
        ax.axis('off')
        fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.cfg.VISUALIZATION_DIR, f"gt_vis_{basename.split('.')[0]}.png")
        plt.savefig(save_path)
        plt.close(fig)


# --- 3. PyTorch Dataset (已增加详细诊断) ---
class WaveStereoDataset(Dataset):
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

        right_frame_basename = frame_basename.replace('lresult', 'rresult')
        right_img_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, right_frame_basename)

        gt_path = os.path.join(self.cfg.GROUND_TRUTH_DIR, f"gt_disparity_{frame_basename.split('.')[0]}.pkl")

        try:
            if not os.path.exists(left_img_path): return None
            left_img = cv2.imread(left_img_path)
            if left_img is None: return None

            if not os.path.exists(right_img_path): return None
            right_img = cv2.imread(right_img_path)
            if right_img is None: return None

            if not os.path.exists(gt_path): return None
            with open(gt_path, 'rb') as f:
                disparity_gt = pickle.load(f)

        except Exception:
            return None

        h, w, _ = left_img.shape
        target_h, target_w = self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH
        left_img = cv2.resize(left_img, (target_w, target_h))
        right_img = cv2.resize(right_img, (target_w, target_h))
        disparity_gt = cv2.resize(disparity_gt, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        disparity_gt *= (target_w / w)
        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float() / 255.0
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float() / 255.0
        disparity_tensor = torch.from_numpy(disparity_gt).float().unsqueeze(0)
        return left_tensor, right_tensor, disparity_tensor


# --- 4. 深度学习模型 (已修正) ---
class DINOv3StereoModel(nn.Module):
    def __init__(self, cfg: Config, out_channels=1):
        super().__init__()
        self.cfg = cfg

        self.dino = self._load_dinov3_model()
        if self.dino is None:
            raise RuntimeError("无法加载DINOv3模型，请检查模型文件或网络连接")

        for param in self.dino.parameters():
            param.requires_grad = False

        self.feature_dim = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size
        self.num_register_tokens = self.dino.config.num_register_tokens

        print(f"模型特征维度: {self.feature_dim}, Patch Size: {self.patch_size}")
        print(f"检测到 {self.num_register_tokens} 个 Register Tokens。")

        self.decoder = nn.Sequential(
            nn.Conv2d(self.feature_dim * 2, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        print("模型构建完成。")

    def _load_dinov3_model(self):
        model = None

        if self.cfg.check_model_files():
            try:
                print(f"尝试从本地路径加载模型: {self.cfg.DINO_MODEL_NAME}")
                model = AutoModel.from_pretrained(
                    self.cfg.DINO_MODEL_NAME,
                    local_files_only=True
                )
                print("✓ 成功从本地加载DINOv3模型")
                return model
            except Exception as e:
                print(f"[!] 从本地加载模型失败: {e}")

        try:
            print(f"尝试从Hugging Face Hub加载模型: {self.cfg.DINO_ONLINE_MODEL}")

            print("--> 正在从 Hub 加载模型配置...")
            config = AutoConfig.from_pretrained(
                self.cfg.DINO_ONLINE_MODEL,
                local_files_only=False
            )

            print("--> 正在使用该配置加载模型...")
            model = AutoModel.from_pretrained(
                self.cfg.DINO_ONLINE_MODEL,
                config=config,
                local_files_only=False
            )

            print("✓ 成功从Hugging Face Hub加载DINOv3模型")

            os.makedirs(self.cfg.DINO_MODEL_NAME, exist_ok=True)
            model.save_pretrained(self.cfg.DINO_MODEL_NAME)
            print(f"模型已保存到本地: {self.cfg.DINO_MODEL_NAME}")
            return model
        except Exception as e:
            print(f"[!] 从Hugging Face Hub加载模型失败: {e}")

        try:
            print("尝试使用最小配置创建DINOv3模型")
            config = Dinov2Config()
            model = Dinov2Model(config)
            print("✓ 成功使用最小配置创建DINOv3模型")
            return model
        except Exception as e:
            print(f"[!] 使用最小配置创建模型失败: {e}")

        return model

    def forward(self, left_image, right_image):
        b, c, h, w = left_image.shape

        with torch.no_grad():
            outputs_left = self.dino(left_image)
            outputs_right = self.dino(right_image)
            features_left = outputs_left.last_hidden_state
            features_right = outputs_right.last_hidden_state

        start_index = 1 + self.num_register_tokens
        patch_tokens_left = features_left[:, start_index:, :]
        patch_tokens_right = features_right[:, start_index:, :]

        feature_h, feature_w = h // self.patch_size, w // self.patch_size

        expected_num_tokens = feature_h * feature_w
        assert patch_tokens_left.shape[1] == expected_num_tokens
        assert patch_tokens_right.shape[1] == expected_num_tokens

        features_left_2d = patch_tokens_left.permute(0, 2, 1).reshape(b, self.feature_dim, feature_h, feature_w)
        features_right_2d = patch_tokens_right.permute(0, 2, 1).reshape(b, self.feature_dim, feature_h, feature_w)

        x = torch.cat([features_left_2d, features_right_2d], dim=1)

        predicted_disparity_low_res = self.decoder(x)

        predicted_disparity = nn.functional.interpolate(
            predicted_disparity_low_res,
            size=(h, w),
            mode='bilinear',
            align_corners=True
        )

        return predicted_disparity


# --- 5. 训练循环 (已增加诊断代码和None值检查) ---
def train(cfg: Config):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else None

    train_dataset = WaveStereoDataset(cfg, is_validation=False)
    val_dataset = WaveStereoDataset(cfg, is_validation=True)

    if len(train_dataset) == 0:
        print(f"[!] 严重错误: 训练数据集为空！请检查 LEFT_IMAGE_DIR ('{cfg.LEFT_IMAGE_DIR}') 配置。")
        return

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    try:
        model = DINOv3StereoModel(cfg).to(device)
    except Exception as e:
        print(f"[!] 模型初始化失败: {e}")
        return

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LEARNING_RATE)

    print("--- 开始训练模型 ---")
    best_val_loss = float('inf')
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} [训练]")
        for i, data in enumerate(pbar):
            if data is None:
                continue

            left, right, gt_disp = data
            left, right, gt_disp = left.to(device), right.to(device), gt_disp.to(device)

            if i == 0 and epoch == 0:
                print("\n--- 诊断信息 ---")
                print(f"真值视差图 (gt_disp) shape: {gt_disp.shape}")
                print(f"gt_disp 最小值: {torch.min(gt_disp)}")
                print(f"gt_disp 最大值: {torch.max(gt_disp)}")
                print(f"gt_disp 平均值: {torch.mean(gt_disp)}")

            mask = (gt_disp > 0) & (gt_disp < 1000)

            if i == 0 and epoch == 0:
                valid_pixels = torch.sum(mask)
                print(f"有效像素数量 (Mask Sum): {valid_pixels}")
                if valid_pixels == 0:
                    print("[!] 严重警告: Mask为空，此batch无法计算损失！")
                print("--- 诊断结束 ---\n")

            optimizer.zero_grad()
            pred_disp = model(left, right)

            if torch.sum(mask) == 0: continue

            loss = criterion(pred_disp[mask], gt_disp[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} [验证]"):
                if data is None: continue
                left, right, gt_disp = data
                left, right, gt_disp = left.to(device), right.to(device), gt_disp.to(device)
                pred_disp = model(left, right)
                mask = (gt_disp > 0) & (gt_disp < 1000)
                if torch.sum(mask) == 0: continue
                loss = criterion(pred_disp[mask], gt_disp[mask])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} -> 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"验证损失提升，模型已保存至: {save_path}")


# --- 6. 主执行模块 (已修正为自动化流程) ---
if __name__ == '__main__':
    # 确保在使用 'Agg' 后端时 matplotlib 不会尝试使用 GUI
    if sys.platform.startswith('linux'):
        # 在Linux服务器上，可能需要设置此环境变量
        os.environ['MPLBACKEND'] = 'Agg'
    plt.switch_backend('Agg')

    config = Config()

    # --- 步骤 1: 自动检查并生成真值数据 ---
    num_left_images = len(glob.glob(os.path.join(config.LEFT_IMAGE_DIR, "*.*")))
    num_gt_files = len(glob.glob(os.path.join(config.GROUND_TRUTH_DIR, "*.pkl")))

    # 如果真值文件数量明显少于图片数量，则运行生成器
    if num_gt_files < num_left_images * 0.9:  # 留一些余量
        print("[!] 检测到真值文件不完整或缺失，开始自动生成...")
        gt_generator = GroundTruthGenerator(config)
        gt_generator.process_all_frames()
    else:
        print("[✓] 真值文件检查通过，跳过生成步骤。")

    # --- 步骤 2: 运行训练 ---
    print("\n" + "=" * 50)
    print("开始模型训练...")
    print("=" * 50)

    # 检查模型文件
    config.check_model_files()

    train(config)
