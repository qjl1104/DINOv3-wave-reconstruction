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
    NUM_ATTENTION_LAYERS: int = 6
    NUM_HEADS: int = 8
    MATCHING_TEMPERATURE: float = 15.0
    EPIPOLAR_THRESHOLD: float = 3.0   # 推理时极线约束阈值 (像素)

    # ===== 训练参数 =====
    BATCH_SIZE: int = 1
    ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 2e-4
    NUM_EPOCHS: int = 150
    SEED: int = 42

    # ===== 损失权重 =====
    PHOTOMETRIC_WEIGHT: float = 5.0
    EPIPOLAR_WEIGHT: float = 0.1
    PHY_SMOOTH_WEIGHT: float = 2.0
    PHY_SLOPE_WEIGHT: float = 0.1
    PHY_ZEROMEAN_WEIGHT: float = 0.1
    PATCH_SIZE_PHOTOMETRIC: int = 11

    # ===== 可视化/日志 =====
    VISUALIZE_INTERVAL: int = 5

    # ===== 推理参数 =====
    CONF_THRESH: float = 0.2

    # ===== 时序分析 =====
    FPS: int = 50
