# DINOv3/batch_pointcloud.py
"""
批量逐帧推理：用 feature_cache（预计算的 DINOv3 特征 + blob 关键点，
跳过 backbone，只跑 proj + 相关 + Sinkhorn）对全部 1000 帧生成 3D 点云，
保存为 pkl 供时空相干性诊断（diag_dinov3_coherence.py）。

checkpoint 用 evaluation_report.json 建档的 20260713-160907/best_model.pth
（300 epoch 收敛、指标有文档；20260716 的 run 是新损失组合的未评估实验）。

过滤流程与 inference.py 一致：scores>0 & disp>10 → 深度 2000..15000mm
→ Z 向 2×IQR。不做 RANSAC 平面减法（下游诊断自己做 PCA）。

用法：../.venv_fs/Scripts/python.exe batch_pointcloud.py
输出：pointclouds_1000f.pkl —— dict: frame_idx(0..999) → (N,3) float64 (X,Y,Z mm)
"""

import glob
import os
import pickle
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config  # noqa: E402
from models import CorrMatchingStereoModel  # noqa: E402
from utils import load_model_checkpoint, reproject_to_3d  # noqa: E402

CKPT = "training_runs/20260713-160907/checkpoints/best_model.pth"
OUT_PKL = "pointclouds_1000f.pkl"


def main():
    cfg = Config()
    calib = np.load(cfg.CALIBRATION_FILE)
    # 关键：dataset.py 构造时会按标定图尺寸回填 cfg.IMAGE_WIDTH/HEIGHT，
    # 几何指纹要除以图像对角线；不建 dataset 直接推理时默认为 0 → 除零
    # → 指纹 NaN → 全帧视差 NaN（本脚本曾因此 valid=0）。
    if cfg.IMAGE_WIDTH == 0:
        cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT = calib["map1_left"].shape[1::-1]
    Q = calib["Q"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型 {CKPT}")
    model = CorrMatchingStereoModel(cfg).to(device)
    load_model_checkpoint(model, CKPT, device)
    model.eval()

    files = sorted(glob.glob("feature_cache/left*.pt"),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    print(f"缓存帧数: {len(files)}")

    clouds = {}
    n_pts = []
    t0 = time.time()
    for fi, fp in enumerate(files):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        lg = d["left_gray"].float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        rg = d["right_gray"].float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        lrgb = lg.repeat(1, 3, 1, 1)
        rrgb = rg.repeat(1, 3, 1, 1)
        mask = torch.ones(1, 1, lg.shape[-2], lg.shape[-1], device=device)
        cached = {"feat_left": d["feat_left"].unsqueeze(0).to(device),
                  "feat_right": d["feat_right"].unsqueeze(0).to(device),
                  "keypoints_left": d["keypoints_left"].unsqueeze(0).to(device),
                  "scores_left": d["scores_left"].unsqueeze(0).to(device),
                  "keypoints_right": d["keypoints_right"].unsqueeze(0).to(device),
                  "scores_right": d["scores_right"].unsqueeze(0).to(device)}
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(device.type == "cuda")):
            out = model(lg, rg, lrgb, rrgb, mask,
                        cached_data=cached, reverse_match=False)

        kpl = out["keypoints_left"][0].float().cpu().numpy()
        scores = out["scores_left"][0].float().cpu().numpy()
        disp = out["disparity"][0].float().cpu().numpy()
        valid = (scores > 0) & (disp > 10.0)
        pts = np.zeros((0, 3))
        if valid.sum() >= 10:
            pts = reproject_to_3d(kpl[valid], disp[valid], Q).astype(np.float64)
            Z = pts[:, 2]
            pts = pts[(Z > 2000) & (Z < 15000)]
            if len(pts) >= 10:
                zm = np.median(pts[:, 2])
                ziqr = np.percentile(pts[:, 2], 75) - np.percentile(pts[:, 2], 25)
                pts = pts[np.abs(pts[:, 2] - zm) < 2.0 * max(ziqr, 1e-6)]
        clouds[fi] = pts
        n_pts.append(len(pts))
        if (fi + 1) % 100 == 0:
            print(f"{fi + 1}/{len(files)}  用时 {time.time() - t0:.0f}s  "
                  f"近100帧平均点数 {np.mean(n_pts[-100:]):.0f}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(clouds, f)
    print(f"[输出] {OUT_PKL} | 空帧 {sum(1 for v in clouds.values() if len(v) == 0)} 个 | "
          f"总点数 {sum(len(v) for v in clouds.values())}")


if __name__ == "__main__":
    main()
