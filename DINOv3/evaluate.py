"""
全验证集量化评估脚本
======================
评估指标：
  1. 极线误差统计（均值、标准差、<0.5px/<1px/<3px 占比）
  2. 物理精度（波高 MAE vs GT 40mm, 波长 MAE vs GT 2500mm）
  3. 时序一致性（帧间视差 jitter 和高度 jitter）
  4. Left-Right 一致性检查
  5. 推理速度（ms/frame）
  6. 与 NCC baseline 对比

用法：
    python evaluate.py --checkpoint path/to/best_model.pth
    python evaluate.py --checkpoint path/to/best_model.pth --max_frames 20
"""

import os
import sys
import json
import argparse
import time
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models import CorrMatchingStereoModel
from dataset import RectifiedWaveStereoDataset, stereo_collate_fn
from utils import reproject_to_3d, load_model_checkpoint


# ============================================================
# 正弦波函数和拟合（复用自 inference.py）
# ============================================================

def wave_func(z, A, k, phi, offset):
    return A * np.cos(k * z + phi) + offset


def fit_wave_relaxed(z, y):
    k_init = 2 * np.pi / 2500
    A_init = (np.max(y) - np.min(y)) / 2
    p0 = [A_init, k_init, 0, np.mean(y)]
    try:
        popt, _ = curve_fit(wave_func, z, y, p0=p0, maxfev=10000)
        return popt, abs(popt[0]) * 2
    except Exception:
        return [0, 0, 0, 0], 0


# ============================================================
# NCC 匹配（复用自 ablation_sparse.py）
# ============================================================

def ncc_match_keypoints(left_gray_np, right_gray_np, keypoints_left,
                        patch_size=21, search_range=1000):
    half = patch_size // 2
    H, W = left_gray_np.shape
    N = len(keypoints_left)
    disparities = np.zeros(N)
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        x_l, y_l = int(keypoints_left[i, 0]), int(keypoints_left[i, 1])
        if (y_l - half < 0 or y_l + half >= H or
            x_l - half < 0 or x_l + half >= W):
            continue
        patch_l = left_gray_np[y_l - half:y_l + half + 1,
                               x_l - half:x_l + half + 1].astype(np.float32)
        if patch_l.std() < 1.0:
            continue
        x_start = max(half, x_l - search_range)
        x_end = max(half, x_l - 1)
        if x_start >= x_end:
            continue
        search_region = right_gray_np[y_l - half:y_l + half + 1,
                                      x_start - half:x_end + half + 1].astype(np.float32)
        if search_region.shape[1] < patch_l.shape[1]:
            continue
        res = cv2.matchTemplate(search_region, patch_l, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.3:
            best_x = x_start + max_loc[0]
            disparities[i] = x_l - best_x
            valid_mask[i] = True
    return disparities, valid_mask


# ============================================================
# RANSAC 平面拟合（复用自 inference.py）
# ============================================================

def fit_plane_ransac(pts_3d, residual_thresh=30.0, min_samples=3):
    X_feat = pts_3d[:, [0, 2]]
    Y_target = pts_3d[:, 1]
    ransac = RANSACRegressor(residual_threshold=residual_thresh,
                             min_samples=min_samples, random_state=42)
    ransac.fit(X_feat, Y_target)
    Y_plane = ransac.predict(X_feat)
    inlier_mask = ransac.inlier_mask_
    return Y_plane, inlier_mask


# ============================================================
# 从模型输出中提取有效关键点
# ============================================================

def get_valid_kp(out, min_disp=1.0):
    kpl = out['keypoints_left'][0].cpu().numpy()
    kpr_pred = out['keypoints_right_pred'][0].cpu().numpy()
    scores = out['scores_left'][0].cpu().numpy()
    disp = out['disparity'][0].cpu().numpy()

    valid = (scores > 0) & (disp > min_disp)
    return kpl[valid], kpr_pred[valid], disp[valid], scores[valid]


# ============================================================
# 3D 重建 + 滤波 + 波拟合（单帧）
# ============================================================

def reconstruct_and_fit(kp_valid, disp_valid, Q_np, depth_min=2000, depth_max=15000):
    if len(kp_valid) < 10:
        return None

    pts_3d = reproject_to_3d(kp_valid, disp_valid, Q_np)
    if len(pts_3d) < 10:
        return None

    Z = pts_3d[:, 2]
    mask_z = (Z > depth_min) & (Z < depth_max)
    pts_filt = pts_3d[mask_z]
    kp_filt = kp_valid[mask_z]
    if len(pts_filt) < 10:
        return None

    z_median = np.median(pts_filt[:, 2])
    z_iqr = np.percentile(pts_filt[:, 2], 75) - np.percentile(pts_filt[:, 2], 25)
    z_mask = np.abs(pts_filt[:, 2] - z_median) < 2.0 * z_iqr
    pts_filt = pts_filt[z_mask]
    kp_filt = kp_filt[z_mask]
    if len(pts_filt) < 10:
        return None

    Y_plane, inlier_mask = fit_plane_ransac(pts_filt, residual_thresh=25.0)
    wave_height = pts_filt[:, 1] - Y_plane

    X_in = pts_filt[inlier_mask, 0]
    Z_in = pts_filt[inlier_mask, 2]
    wh_in = wave_height[inlier_mask]
    if len(Z_in) < 10:
        return None

    idx_sort = np.argsort(Z_in)
    popt, H = fit_wave_relaxed(Z_in[idx_sort], wh_in[idx_sort])
    wavelength = 2 * np.pi / abs(popt[1]) if abs(popt[1]) > 1e-10 else 0

    return {
        'height': H,
        'wavelength': wavelength,
        'height_std': float(np.std(wh_in)),
        'height_range': float(np.max(wh_in) - np.min(wh_in)),
        'n_points': len(Z_in),
        'z_in': Z_in,
        'wh_in': wh_in,
        'popt': popt,
    }


# ============================================================
# Left-Right 一致性检查
# ============================================================

def compute_lr_consistency(kpr_pred, kpr_actual, sr_actual, threshold=1.0):
    valid_right = sr_actual > 0
    kpr_v = kpr_actual[valid_right]
    if len(kpr_pred) == 0 or len(kpr_v) == 0:
        return 0.0

    distances = np.sqrt(
        (kpr_pred[:, None, 0] - kpr_v[None, :, 0]) ** 2 +
        (kpr_pred[:, None, 1] - kpr_v[None, :, 1]) ** 2
    )
    min_dist = distances.min(axis=1)
    return float((min_dist < threshold).mean())


# ============================================================
# 主评估函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="全验证集量化评估")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备")
    parser.add_argument("--output", type=str, default="evaluation_report.json",
                        help="报告输出 JSON 路径")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="最大评估帧数 (0=全部)")
    parser.add_argument("--max_temporal", type=int, default=50,
                        help="时序分析最大帧数")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Config()
    use_cached = cfg.USE_FEATURE_CACHE

    # ---- 加载模型 ----
    print(f"\n[加载] Checkpoint: {args.checkpoint}")
    model = CorrMatchingStereoModel(cfg).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    model.eval()

    # ---- 验证数据集 ----
    val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=stereo_collate_fn, num_workers=0)

    total_frames = len(val_ds)
    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    max_frames = min(max_frames, total_frames)
    max_temporal = min(args.max_temporal, max_frames)

    print(f"\n验证集帧数: {total_frames}, 评估帧数: {max_frames}, "
          f"时序帧数: {max_temporal}, 缓存模式: {use_cached}")

    # ---- 逐帧存储 ----
    epi_errors_all = []
    heights_all = []
    wavelengths_all = []
    lr_inlier_rates = []
    disp_per_frame = []
    height_std_per_frame = []
    inference_times = []
    frame_kp_counts = []

    ncc_epi_errors_all = []
    ncc_heights_all = []
    ncc_wavelengths_all = []
    ncc_lr_inlier_rates = []
    ncc_disp_per_frame = []
    ncc_height_std_per_frame = []

    temporal_disps = []
    temporal_heights = []
    temporal_height_stds = []

    print(f"\n{'=' * 60}")
    print(f"  开始逐帧评估...")
    print(f"{'=' * 60}\n")

    processed = 0

    for _, batch in enumerate(tqdm(val_loader, desc="评估进度", unit="帧")):
        if batch is None:
            continue
        if processed >= max_frames:
            break

        # ---- 准备数据 ----
        lg = batch['left_gray'].to(device)
        rg = batch['right_gray'].to(device)
        mask = batch['mask'].to(device)
        Q_np = batch['Q'][0].cpu().numpy()

        if use_cached and batch.get('cached', False):
            lrgb = lg.repeat(1, 3, 1, 1)
            rrgb = rg.repeat(1, 3, 1, 1)
            cached_data = {
                'feat_left': batch['feat_left'].to(device),
                'feat_right': batch['feat_right'].to(device),
                'keypoints_left': batch['keypoints_left'].to(device),
                'scores_left': batch['scores_left'].to(device),
                'keypoints_right': batch['keypoints_right'].to(device),
                'scores_right': batch['scores_right'].to(device),
            }
        else:
            lrgb = batch['left_rgb'].to(device)
            rrgb = batch['right_rgb'].to(device)
            cached_data = None

        # ---- 模型推理 + 计时 ----
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                out = model(lg, rg, lrgb, rrgb, mask, cached_data=cached_data)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        inference_times.append((t1 - t0) * 1000)

        # ---- 提取有效关键点 ----
        kpl_v, kpr_pred_v, disp_v, scores_v = get_valid_kp(out, min_disp=1.0)

        if len(kpl_v) < 3:
            continue

        frame_kp_counts.append(len(kpl_v))

        # ---- 1. 极线误差 ----
        epi_err = np.abs(kpl_v[:, 1] - kpr_pred_v[:, 1])
        epi_errors_all.append({
            'mean': float(epi_err.mean()),
            'std': float(epi_err.std()),
            'lt_05': float((epi_err < 0.5).mean()),
            'lt_1': float((epi_err < 1.0).mean()),
            'lt_3': float((epi_err < 3.0).mean()),
            'values': epi_err.tolist(),
            'n_kp': int(len(kpl_v)),
        })

        # ---- 2. 视差 ----
        mean_disp = float(disp_v.mean())
        disp_per_frame.append(mean_disp)

        # ---- 3. 3D 重建 + 波拟合 ----
        recon = reconstruct_and_fit(kpl_v, disp_v, Q_np)
        if recon is not None and recon['height'] > 0.1:
            heights_all.append(recon['height'])
            wavelengths_all.append(recon['wavelength'])
            height_std_per_frame.append(recon['height_std'])

            if processed < max_temporal:
                temporal_disps.append(mean_disp)
                temporal_heights.append(recon['height'])
                temporal_height_stds.append(recon['height_std'])
        elif processed < max_temporal:
            if len(temporal_disps) > 0:
                temporal_disps.append(mean_disp)
                temporal_heights.append(temporal_heights[-1] if temporal_heights else 0)
                temporal_height_stds.append(temporal_height_stds[-1] if temporal_height_stds else 0)
            else:
                temporal_disps.append(mean_disp)
                temporal_heights.append(0)
                temporal_height_stds.append(0)

        # ---- 4. Left-Right 一致性 ----
        kpr_actual = out['keypoints_right'][0].cpu().numpy()
        sr_actual = out['scores_right'][0].cpu().numpy()
        lr_rate = compute_lr_consistency(kpr_pred_v, kpr_actual, sr_actual)
        lr_inlier_rates.append(lr_rate)

        # ---- 5. NCC Baseline ----
        l_np = (lg[0, 0].cpu().numpy() * 255).astype(np.uint8)
        r_np = (rg[0, 0].cpu().numpy() * 255).astype(np.uint8)

        ncc_disps, ncc_valid = ncc_match_keypoints(l_np, r_np, kpl_v)
        ncc_n_matched = ncc_valid.sum()

        if ncc_n_matched >= 5:
            ncc_disp_matched = ncc_disps[ncc_valid]
            ncc_kp_matched = kpl_v[ncc_valid]

            ncc_epi_errors_all.append(0.0)
            ncc_disp_per_frame.append(float(np.mean(np.abs(ncc_disp_matched))))

            ncc_kpr_pred = np.column_stack([
                ncc_kp_matched[:, 0] - ncc_disp_matched,
                ncc_kp_matched[:, 1],
            ])
            ncc_lr = compute_lr_consistency(ncc_kpr_pred, kpr_actual, sr_actual)
            ncc_lr_inlier_rates.append(ncc_lr)

            ncc_recon = reconstruct_and_fit(ncc_kp_matched, ncc_disp_matched, Q_np)
            if ncc_recon is not None and ncc_recon['height'] > 0.1:
                ncc_heights_all.append(ncc_recon['height'])
                ncc_wavelengths_all.append(ncc_recon['wavelength'])
                ncc_height_std_per_frame.append(ncc_recon['height_std'])

        processed += 1

    print(f"\n处理完成: {processed} 帧")

    # ============================================================
    # 汇总统计
    # ============================================================

    # 展平所有极线误差
    all_epi_values = []
    for e in epi_errors_all:
        all_epi_values.extend(e['values'])
    all_epi_values = np.array(all_epi_values)

    epi_mean = float(np.mean(all_epi_values)) if len(all_epi_values) > 0 else 0
    epi_std = float(np.std(all_epi_values)) if len(all_epi_values) > 0 else 0
    epi_lt_05 = float((all_epi_values < 0.5).mean()) if len(all_epi_values) > 0 else 0
    epi_lt_1 = float((all_epi_values < 1.0).mean()) if len(all_epi_values) > 0 else 0
    epi_lt_3 = float((all_epi_values < 3.0).mean()) if len(all_epi_values) > 0 else 0

    heights_arr = np.array(heights_all) if heights_all else np.array([])
    wavelengths_arr = np.array(wavelengths_all) if wavelengths_all else np.array([])
    lr_arr = np.array(lr_inlier_rates) if lr_inlier_rates else np.array([])
    disp_arr = np.array(disp_per_frame) if disp_per_frame else np.array([])
    hs_arr = np.array(height_std_per_frame) if height_std_per_frame else np.array([])
    time_arr = np.array(inference_times) if inference_times else np.array([])

    ncc_heights_arr = np.array(ncc_heights_all) if ncc_heights_all else np.array([])
    ncc_wavelengths_arr = np.array(ncc_wavelengths_all) if ncc_wavelengths_all else np.array([])
    ncc_lr_arr = np.array(ncc_lr_inlier_rates) if ncc_lr_inlier_rates else np.array([])

    # 物理 GT（造波机参数）
    GT_HEIGHT = 40.0
    GT_WAVELENGTH = 2500.0

    height_mae = float(np.mean(np.abs(heights_arr - GT_HEIGHT))) if len(heights_arr) > 0 else None
    wavelength_mae = float(np.mean(np.abs(wavelengths_arr - GT_WAVELENGTH))) if len(wavelengths_arr) > 0 else None
    height_mae_pct = height_mae / GT_HEIGHT * 100 if height_mae is not None else None
    wavelength_mae_pct = wavelength_mae / GT_WAVELENGTH * 100 if wavelength_mae is not None else None

    ncc_height_mae = float(np.mean(np.abs(ncc_heights_arr - GT_HEIGHT))) if len(ncc_heights_arr) > 0 else None
    ncc_wavelength_mae = float(np.mean(np.abs(ncc_wavelengths_arr - GT_WAVELENGTH))) if len(ncc_wavelengths_arr) > 0 else None
    ncc_height_mae_pct = ncc_height_mae / GT_HEIGHT * 100 if ncc_height_mae is not None else None
    ncc_wavelength_mae_pct = ncc_wavelength_mae / GT_WAVELENGTH * 100 if ncc_wavelength_mae is not None else None

    # 时序 jitter
    if len(temporal_disps) >= 2:
        td = np.array(temporal_disps)
        disp_jitter = float(np.mean(np.abs(np.diff(td))))
    else:
        disp_jitter = None

    if len(temporal_heights) >= 2:
        th = np.array(temporal_heights)
        height_jitter = float(np.mean(np.abs(np.diff(th))))
    else:
        height_jitter = None

    # ============================================================
    # 输出报告
    # ============================================================

    def fmt(val, decimals=1):
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}"

    def fmt_pct(val, decimals=1):
        if val is None:
            return "N/A"
        return f"{val * 100:.{decimals}f}%"

    def fmt_mae(mae, pct, unit, decimals=1):
        if mae is None:
            return "N/A"
        pct_str = f"({pct:.1f}%)" if pct is not None else ""
        return f"{mae:.{decimals}f} {unit}   {pct_str}"

    print(f"\n{'=' * 60}")
    print(f"  Evaluation Report")
    print(f"{'=' * 60}")

    print(f"\n[Geometry] 极线误差")
    print(f"  Mean ± Std:          {epi_mean:.2f} ± {epi_std:.2f} px")
    print(f"  <0.5px ratio:        {fmt_pct(epi_lt_05, 1)}")
    print(f"  <1.0px ratio:        {fmt_pct(epi_lt_1, 1)}")
    print(f"  <3.0px ratio:        {fmt_pct(epi_lt_3, 1)}")
    print(f"  Total keypoints:     {len(all_epi_values)}")

    print(f"\n[Physics] 物理精度 (GT: H=40mm, λ≈2500mm)")
    if len(heights_arr) > 0:
        print(f"  Wave Height MAE:     {fmt_mae(height_mae, height_mae_pct, 'mm')}")
        print(f"  Wavelength MAE:      {fmt_mae(wavelength_mae, wavelength_mae_pct, 'mm')}")
        print(f"  Height Std Dev:      {fmt(hs_arr.mean(), 1)} mm")
        print(f"  Fitted frames:       {len(heights_arr)}")
    else:
        print(f"  Wave Height MAE:     N/A (无有效帧)")
        print(f"  Wavelength MAE:      N/A")
        print(f"  Height Std Dev:      N/A")
        print(f"  Fitted frames:       0")

    print(f"\n[Consistency] Left-Right 一致性")
    lr_mean = float(lr_arr.mean()) if len(lr_arr) > 0 else None
    print(f"  LR <1px inlier:      {fmt_pct(lr_mean)}")

    print(f"\n[Temporal] 时序稳定性")
    print(f"  Disparity Jitter:    {fmt(disp_jitter, 2)} px/frame")
    print(f"  Height Jitter:       {fmt(height_jitter, 1)} mm/frame")
    print(f"  Analyzed frames:     {len(temporal_disps) if temporal_disps else 0}")

    print(f"\n[Speed]")
    if len(time_arr) > 0:
        print(f"  Avg Inference:       {fmt(time_arr.mean(), 1)} ms/frame")
        print(f"  Std Dev:             {fmt(time_arr.std(), 1)} ms")
        print(f"  Min / Max:           {fmt(time_arr.min(), 1)} / {fmt(time_arr.max(), 1)} ms")

    print(f"\n[vs NCC]")
    if len(ncc_heights_arr) > 0:
        print(f"  极线误差:    {epi_mean:.2f} vs 0.00* px")
        print(f"  LR一致性:    {fmt_pct(lr_mean)} vs {fmt_pct(float(ncc_lr_arr.mean()) if len(ncc_lr_arr) > 0 else None)}")
        print(f"  高度MAE:     {fmt_mae(height_mae, height_mae_pct, 'mm')} vs {fmt_mae(ncc_height_mae, ncc_height_mae_pct, 'mm')}")
        print(f"  波长MAE:     {fmt_mae(wavelength_mae, wavelength_mae_pct, 'mm')} vs {fmt_mae(ncc_wavelength_mae, ncc_wavelength_mae_pct, 'mm')}")
    else:
        print(f"  极线误差:    {epi_mean:.2f} vs 0.00* px")
        print(f"  LR一致性:    {fmt_pct(lr_mean)} vs N/A")
        print(f"  高度MAE:     {fmt_mae(height_mae, height_mae_pct, 'mm')} vs N/A")
        print(f"  波长MAE:     {fmt_mae(wavelength_mae, wavelength_mae_pct, 'mm')} vs N/A")

    print(f"\n[Info]")
    print(f"  评估帧数:            {processed}")
    print(f"  平均关键点数:        {fmt(np.mean(frame_kp_counts) if frame_kp_counts else 0, 0)}")
    print(f"  * NCC 极线误差始终为 0（沿极线搜索）")

    print(f"\n{'=' * 60}")

    # ============================================================
    # 保存 JSON 报告
    # ============================================================

    report = {
        'config': {
            'checkpoint': args.checkpoint,
            'device': str(device),
            'frames_evaluated': processed,
            'frames_total': total_frames,
            'cached_mode': use_cached,
        },
        'epipolar_error': {
            'mean_px': epi_mean,
            'std_px': epi_std,
            'lt_0.5px_ratio': epi_lt_05,
            'lt_1.0px_ratio': epi_lt_1,
            'lt_3.0px_ratio': epi_lt_3,
            'total_keypoints': int(len(all_epi_values)),
        },
        'physics': {
            'gt_height_mm': GT_HEIGHT,
            'gt_wavelength_mm': GT_WAVELENGTH,
            'height_mae_mm': height_mae,
            'height_mae_pct': height_mae_pct,
            'wavelength_mae_mm': wavelength_mae,
            'wavelength_mae_pct': wavelength_mae_pct,
            'height_std_mean_mm': float(hs_arr.mean()) if len(hs_arr) > 0 else None,
            'fitted_frames': len(heights_arr),
        },
        'lr_consistency': {
            'lt_1px_inlier_ratio': lr_mean,
            'frames': len(lr_inlier_rates),
        },
        'temporal': {
            'disparity_jitter_px_per_frame': disp_jitter,
            'height_jitter_mm_per_frame': height_jitter,
            'analyzed_frames': len(temporal_disps),
        },
        'speed': {
            'avg_ms': float(time_arr.mean()) if len(time_arr) > 0 else None,
            'std_ms': float(time_arr.std()) if len(time_arr) > 0 else None,
            'min_ms': float(time_arr.min()) if len(time_arr) > 0 else None,
            'max_ms': float(time_arr.max()) if len(time_arr) > 0 else None,
        },
        'ncc_baseline': {
            'height_mae_mm': ncc_height_mae,
            'height_mae_pct': ncc_height_mae_pct,
            'wavelength_mae_mm': ncc_wavelength_mae,
            'wavelength_mae_pct': ncc_wavelength_mae_pct,
            'lr_inlier_ratio': float(ncc_lr_arr.mean()) if len(ncc_lr_arr) > 0 else None,
            'note': 'NCC 极线误差始终为 0（沿极线搜索）',
        },
        'per_frame': {
            'heights': heights_all,
            'wavelengths': wavelengths_all,
            'height_stds': height_std_per_frame,
            'lr_inlier_rates': lr_inlier_rates,
            'disparities': disp_per_frame,
            'inference_times_ms': inference_times,
        },
    }

    output_path = args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()