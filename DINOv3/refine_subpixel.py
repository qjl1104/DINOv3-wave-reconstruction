# DINOv3/refine_subpixel.py
"""
修复1：DINOv3 逐帧匹配的亚像素细化。

诊断（diag_dinov3_coherence.py）定位的病根：相关体是 16px patch 分辨率，
soft-argmax 期望回归的视差抖动 ~7px，而 Z≈9.8m 处每 ±1px 视差误差 =
±34mm 深度噪声（波幅才 40mm）——信号被淹。

修复策略（coarse-to-fine，立体匹配标准做法）：
DINOv3 的 coarse 匹配只负责"消歧"（重复泡沫点里找对那个），精对齐：
1. snap：把预测的右图位置吸附到最近的右图 blob 质心（OpenCV 亚像素），
   约束 |dy|<DY_MAX（极线）且 |x_r − x_pred|<SNAP_R（半 patch 内）；
2. NCC 复核：左图 9×9 patch 对右图 snap 位置 ±4px 一维搜索
   （TM_CCOEFF_NORMED + 抛物线亚像素插值），峰值 <0.5 或落在搜索边缘
   （说明 snap 错了）则丢弃该匹配。

输出 pointclouds_1000f_refined.pkl（同 batch_pointcloud 格式）+
refine_stats.npz（QA 统计）。
用法：../.venv_fs/Scripts/python.exe refine_subpixel.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "training_runs/20260713-160907/checkpoints/best_model.pth"
OUT_PKL = "pointclouds_1000f_refined.pkl"
OUT_STATS = "refine_stats.npz"


# checkpoint 键核对由 utils.load_model_checkpoint 统一报告（缺失/多余键 + proj/geo_fusion 警告）

SNAP_R = 8.0      # snap 搜索半径 px（半 patch）
DY_MAX = 3.0      # 极线 |dy| 上限 px
NCC_HALF = 4      # NCC 一维搜索范围 ±px
NCC_PATCH = 4     # NCC patch 半径（9×9）
NCC_MIN = 0.5     # NCC 峰值下限


def ncc_1d(lgray, rgray, xl, yl, xr, yr):
    """左 patch 对右 (xr±NCC_HALF, yr) 一维 NCC，抛物线插值亚像素偏移。
    返回 (offset, score)；边缘峰值/分数不足/越界返回 None。"""
    h, w = lgray.shape
    p = NCC_PATCH
    if not (p <= xl < w - p and p <= yl < h - p and
            p + NCC_HALF <= xr < w - p - NCC_HALF and p <= yr < h - p):
        return None
    lp = lgray[yl - p:yl + p + 1, xl - p:xl + p + 1].astype(np.float64)
    lp = lp - lp.mean()
    ls = lp.std()
    if ls < 1e-6:
        return None
    scores = np.empty(2 * NCC_HALF + 1)
    for i, dx in enumerate(range(-NCC_HALF, NCC_HALF + 1)):
        rp = rgray[yr - p:yr + p + 1, xr + dx - p:xr + dx + p + 1].astype(np.float64)
        rp = rp - rp.mean()
        rs = rp.std()
        scores[i] = (lp * rp).mean() / (ls * rs) if rs > 1e-6 else -1
    k = int(np.argmax(scores))
    if scores[k] < NCC_MIN or k == 0 or k == len(scores) - 1:
        return None
    y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
    denom = (y0 - 2 * y1 + y2)
    sub = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    return (k - NCC_HALF) + float(np.clip(sub, -1, 1)), float(y1)


def main():
    cfg = Config()
    calib = np.load(cfg.CALIBRATION_FILE)
    if cfg.IMAGE_WIDTH == 0:  # 同 batch_pointcloud：几何指纹除零坑
        cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT = calib["map1_left"].shape[1::-1]
    Q = calib["Q"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型 {CKPT}")
    model = CorrMatchingStereoModel(cfg).to(device)
    load_model_checkpoint(model, CKPT, device)
    model.eval()

    # 锚定脚本目录：CWD 不对时 glob 会落空，空结果会把共享输出 pkl 覆盖成空
    files = sorted(glob.glob(os.path.join(HERE, "feature_cache/left*.pt")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if not files:
        print(f"[错误] 未找到 {os.path.join(HERE, 'feature_cache/left*.pt')}，"
              "请先运行 precompute_cache.py 生成特征缓存")
        sys.exit(1)
    print(f"缓存帧数: {len(files)}")

    clouds = {}
    stat = {"snap_dist": [], "ncc": [], "n_coarse": [], "n_snap": [],
            "n_amb": [], "n_final": []}
    t0 = time.time()
    for fi, fp in enumerate(files):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        lg = d["left_gray"].float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        rg = d["right_gray"].float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        mask = (d["mask"].float().unsqueeze(0).unsqueeze(0) / 255.0).to(device)
        cached = {k: d[k].unsqueeze(0).to(device)
                  for k in ["feat_left", "feat_right", "keypoints_left",
                             "scores_left", "keypoints_right", "scores_right"]}
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(device.type == "cuda")):
            out = model(lg, rg, lg.repeat(1, 3, 1, 1), rg.repeat(1, 3, 1, 1),
                        mask, cached_data=cached, reverse_match=False)

        kpl = out["keypoints_left"][0].float().cpu().numpy()
        scores = out["scores_left"][0].float().cpu().numpy()
        kp_pred = out["keypoints_right_pred"][0].float().cpu().numpy()
        kpr = cached["keypoints_right"][0].float().cpu().numpy()
        lgray = d["left_gray"].numpy()
        rgray = d["right_gray"].numpy()

        keep_l, keep_d = [], []
        n_coarse = n_snap = n_amb = 0
        for (xl, yl), (xp, yp), sc in zip(kpl, kp_pred, scores):
            if sc <= 0:
                continue
            n_coarse += 1
            cand = np.where((np.abs(kpr[:, 1] - yl) < DY_MAX) &
                            (np.abs(kpr[:, 0] - xp) < SNAP_R))[0]
            if len(cand) == 0:
                continue
            dists = np.abs(kpr[cand, 0] - xp)
            if len(cand) > 1:
                n_amb += 1
            j = cand[np.argmin(dists)]
            n_snap += 1
            stat["snap_dist"].append(dists.min())
            xr, yr = int(round(kpr[j, 0])), int(round(kpr[j, 1]))
            ref = ncc_1d(lgray, rgray, int(round(xl)), int(round(yl)), xr, yr)
            if ref is None:
                continue
            off, nccs = ref
            stat["ncc"].append(nccs)
            keep_l.append((xl, yl))
            keep_d.append(xl - (kpr[j, 0] + off))

        stat["n_coarse"].append(n_coarse)
        stat["n_snap"].append(n_snap)
        stat["n_amb"].append(n_amb)
        stat["n_final"].append(len(keep_d))

        pts = np.zeros((0, 3))
        if len(keep_d) >= 10:
            kp_arr = np.array(keep_l)
            disp_arr = np.array(keep_d)
            pts = reproject_to_3d(kp_arr, disp_arr, Q).astype(np.float64)
            Z = pts[:, 2]
            pts = pts[(Z > 2000) & (Z < 15000)]
            if len(pts) >= 10:
                zm = np.median(pts[:, 2])
                ziqr = np.percentile(pts[:, 2], 75) - np.percentile(pts[:, 2], 25)
                pts = pts[np.abs(pts[:, 2] - zm) < 2.0 * max(ziqr, 1e-6)]
        clouds[fi] = pts
        if (fi + 1) % 100 == 0:
            print(f"{fi + 1}/{len(files)}  用时 {time.time() - t0:.0f}s  "
                  f"近100帧平均保留 {np.mean(stat['n_final'][-100:]):.0f} 点")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(clouds, f)
    np.savez(OUT_STATS, **{k: np.array(v) for k, v in stat.items()})
    print(f"[输出] {OUT_PKL} | 总保留点数 {sum(len(v) for v in clouds.values())}")
    sd = np.array(stat["snap_dist"])
    ncc = np.array(stat["ncc"])
    print(f"[统计] snap 距离中位 {np.median(sd):.1f}px | 多候选率 "
          f"{np.sum(stat['n_amb']) / max(np.sum(stat['n_snap']), 1):.1%} | "
          f"NCC 分数中位 {np.median(ncc):.3f} | "
          f"coarse→snap→final 平均 {np.mean(stat['n_coarse']):.0f}→"
          f"{np.mean(stat['n_snap']):.0f}→{np.mean(stat['n_final']):.0f}")


if __name__ == "__main__":
    main()
