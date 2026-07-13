"""
DINOv3 Wave Reconstruction - Unified Configuration
===================================================
All scripts share this single Config dataclass.
"""

import os
import sys
from dataclasses import dataclass

# Project paths (resolved at import time)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(PROJECT_ROOT)


def check_path(path, name):
    """Check if a path exists, exit with error if not."""
    if not os.path.exists(path):
        print(f"\n{'=' * 40}\n[严重错误] 找不到 {name}: {path}\n{'=' * 40}\n")
        sys.exit(1)


@dataclass
class Config:
    # ===== 路径配置 =====
    DINO_LOCAL_PATH: str = os.path.join(PROJECT_ROOT, "dinov3-base-model")
    LEFT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "left_images")
    RIGHT_IMAGE_DIR: str = os.path.join(DATA_ROOT, "data", "right_images")
    CALIBRATION_FILE: str = os.path.join(PROJECT_ROOT, "1128", "paper_params_recalculated.npz")

    # 训练输出目录
    RUNS_BASE_DIR: str = os.path.join(PROJECT_ROOT, "training_runs")

    # 预训练模型 checkpoint (留空则从头训练)
    PRETRAINED_CHECKPOINT: str = ""

    # 推理用 checkpoint (留空则使用未训练权重)
    CHECKPOINT_PATH: str = ""

    # ===== 图像参数 =====
    IMAGE_HEIGHT: int = 0   # 0 = auto-detect from calibration maps
    IMAGE_WIDTH: int = 0
    MASK_THRESHOLD: int = 30

    # ===== Blob 检测参数 =====
    BLOB_MIN_THRESHOLD: float = 15.0
    MAX_KEYPOINTS: int = 1024
    BLOB_MIN_AREA: float = 10.0
    BLOB_MAX_AREA: float = 2500.0

    # ===== 模型架构参数 =====
    FEATURE_DIM: int = 768
    CORR_PROJ_DIM: int = 128
    MATCHING_TEMPERATURE: float = 15.0
    EPIPOLAR_THRESHOLD: float = 3.0

    # ===== 几何指纹参数（DINO + 几何融合） =====
    GEO_KNN_K: int = 8          # 几何指纹的近邻数量
    GEO_FUSION_DIM: int = 128   # 融合后的特征维度（与 CORR_PROJ_DIM 一致）

    # ===== 训练参数 =====
    BATCH_SIZE: int = 4
    ACCUMULATION_STEPS: int = 1
    LEARNING_RATE: float = 5e-5  # 降低 LR 防止几何指纹融合训练发散
    NUM_EPOCHS: int = 300
    SEED: int = 42

    # ===== 损失权重 =====
    CORRELATION_WEIGHT: float = 2.0   # 相关体损失（熵 + 峰度），替代光度损失
    PHOTOMETRIC_WEIGHT: float = 0.0   # 光度损失已废弃（水面不满足亮度恒定假设）
    DISPARITY_WEIGHT: float = 1.0
    PHY_SMOOTH_WEIGHT: float = 1.0
    PHY_SLOPE_WEIGHT: float = 0.3
    PHY_ZEROMEAN_WEIGHT: float = 0.05
    PATCH_SIZE_PHOTOMETRIC: int = 11

    # ===== PINN 物理约束参数 =====
    DEPTH_MIN: float = 100.0
    DEPTH_MAX: float = 30000.0
    SLOPE_THRESHOLD: float = 0.4
    KNN_K: int = 5
    MAX_PINN_POINTS: int = 2000

    # ===== 可视化/日志 =====
    VISUALIZE_INTERVAL: int = 5

    # ===== 推理参数 =====
    CONF_THRESH: float = 0.2

    # ===== 时序分析 =====
    FPS: int = 50

    # ===== 降采样消融实验参数 =====
    KEEP_RATIO: float = 1.0   # 1.0 = 保留所有点, 0.1 = 只保留 10%

    # ===== 特征缓存 =====
    FEATURE_CACHE_DIR: str = os.path.join(PROJECT_ROOT, "feature_cache")
    USE_FEATURE_CACHE: bool = True
    NUM_WORKERS: int = 4
