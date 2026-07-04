"""
DINOv3 模型诊断 — 可视化匹配质量
==================================
对单帧进行推理，绘制：
  1. 左右图上的匹配连线
  2. 视差分布直方图
  3. 极线误差分布
  4. 与 NCC baseline 的视差对比
"""

import os, sys, glob, argparse
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models import CorrMatchingStereoModel, SparseKeypointDetector
from utils import pad_to_patch_size, infer_right_path, load_model_checkpoint



def ncc_match(left_np, right_np, kp_left, patch_size=21, search_range=1000):
    """NCC matching along epipolar line."""
    half = patch_size // 2
    H, W = left_np.shape
    N = len(kp_left)
    disps = np.full(N, np.nan)
    ncc_scores = np.full(N, np.nan)
    
    for i in range(N):
        x_l, y_l = int(kp_left[i, 0]), int(kp_left[i, 1])
        if y_l-half < 0 or y_l+half >= H or x_l-half < 0 or x_l+half >= W:
            continue
        patch_l = left_np[y_l-half:y_l+half+1, x_l-half:x_l+half+1].astype(np.float32)
        ps, pm = patch_l.std(), patch_l.mean()
        if ps < 1: continue
        patch_l_n = (patch_l - pm) / ps
        
        best_ncc, best_x = -1, -1
        for x_r in range(max(half, x_l - search_range), max(half, x_l)):
            if x_r + half >= W: continue
            pr = right_np[y_l-half:y_l+half+1, x_r-half:x_r+half+1].astype(np.float32)
            rs = pr.std()
            if rs < 1: continue
            ncc = np.mean(patch_l_n * (pr - pr.mean()) / rs)
            if ncc > best_ncc:
                best_ncc, best_x = ncc, x_r
        if best_ncc > 0.3 and best_x >= 0:
            disps[i] = x_l - best_x
            ncc_scores[i] = best_ncc
    return disps, ncc_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--output", type=str, default="model_diagnostic.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Load model
    model = CorrMatchingStereoModel(cfg).to(device)
    load_model_checkpoint(model, args.checkpoint, device, strict=False)
    model.eval()

    # Load calibration
    calib = np.load(cfg.CALIBRATION_FILE)
    m1l, m2l = calib['map1_left'], calib['map2_left']
    m1r, m2r = calib['map1_right'], calib['map2_right']
    Q = calib['Q']

    # Find image
    l_files = sorted(glob.glob(os.path.join(cfg.LEFT_IMAGE_DIR, "*.*")))
    idx = min(args.image_index, len(l_files) - 1)
    l_path = l_files[idx]
    r_path = infer_right_path(l_path, cfg.RIGHT_IMAGE_DIR)
    bn = os.path.basename(l_path)
    
    print(f"Diagnosing: {bn}")

    # Rectify
    l_raw = cv2.imread(l_path, 0)
    r_raw = cv2.imread(r_path, 0)
    l_rect = cv2.remap(l_raw, m1l, m2l, cv2.INTER_LINEAR)
    r_rect = cv2.remap(r_raw, m1r, m2r, cv2.INTER_LINEAR)

    # Tensors
    lg = torch.from_numpy(l_rect).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    rg = torch.from_numpy(r_rect).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    lrgb = torch.from_numpy(cv2.cvtColor(l_rect, cv2.COLOR_GRAY2RGB).transpose(2,0,1)).float().unsqueeze(0).to(device) / 255.0
    rrgb = torch.from_numpy(cv2.cvtColor(r_rect, cv2.COLOR_GRAY2RGB).transpose(2,0,1)).float().unsqueeze(0).to(device) / 255.0
    _, mask = cv2.threshold(l_rect, cfg.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    patch_size = model.ext.patch
    inputs = [pad_to_patch_size(t, patch_size=patch_size)[0] for t in [lg, rg, lrgb, rrgb, mask_t]]

    # Model inference
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(*inputs)

    kpl = out['keypoints_left'][0].cpu().numpy()
    kpr = out['keypoints_right_pred'][0].cpu().numpy()
    scores = out['scores_left'][0].cpu().numpy()
    disp_model = out['disparity'][0].cpu().numpy()

    valid = scores > 0
    kpl_v = kpl[valid]
    kpr_v = kpr[valid]
    disp_model_v = disp_model[valid]
    epi_err = np.abs(kpl_v[:, 1] - kpr_v[:, 1])

    # NCC matching
    ncc_disps, ncc_scores = ncc_match(l_rect, r_rect, kpl_v)
    ncc_valid = ~np.isnan(ncc_disps)

    print(f"\n{'='*50}")
    print(f"PI-DINOv3 Model:")
    print(f"  Valid keypoints: {valid.sum()}")
    print(f"  Disparity: {disp_model_v.mean():.1f} ± {disp_model_v.std():.1f}")
    print(f"  Disparity range: [{disp_model_v.min():.1f}, {disp_model_v.max():.1f}]")
    print(f"  Epipolar error: {epi_err.mean():.2f} ± {epi_err.std():.2f} px")
    print(f"  % with EpiErr < 1px: {(epi_err < 1).mean()*100:.1f}%")
    print(f"  % with EpiErr < 3px: {(epi_err < 3).mean()*100:.1f}%")
    print(f"\nNCC Baseline:")
    print(f"  Valid matches: {ncc_valid.sum()} / {len(kpl_v)}")
    ncc_v = ncc_disps[ncc_valid]
    print(f"  Disparity: {ncc_v.mean():.1f} ± {ncc_v.std():.1f}")
    print(f"  Disparity range: [{ncc_v.min():.1f}, {ncc_v.max():.1f}]")
    print(f"{'='*50}")

    # ---- Visualization ----
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (a) Left image with keypoints
    ax = axes[0, 0]
    ax.imshow(l_rect, cmap='gray')
    ax.scatter(kpl_v[:, 0], kpl_v[:, 1], c='lime', s=3, alpha=0.5)
    ax.set_title(f"Left Image ({len(kpl_v)} keypoints)", fontsize=12)
    ax.axis('off')

    # (b) Matching lines (sample 50)
    ax = axes[0, 1]
    combo = np.hstack([l_rect, r_rect])
    ax.imshow(combo, cmap='gray')
    W = l_rect.shape[1]
    sample = np.random.choice(len(kpl_v), min(50, len(kpl_v)), replace=False)
    for i in sample:
        color = 'lime' if epi_err[i] < 3 else 'red'
        ax.plot([kpl_v[i, 0], kpr_v[i, 0] + W], [kpl_v[i, 1], kpr_v[i, 1]], 
                color=color, linewidth=0.5, alpha=0.7)
    ax.set_title("Matching Lines (green=good, red=epi>3px)", fontsize=12)
    ax.axis('off')

    # (c) Epipolar error histogram
    ax = axes[0, 2]
    ax.hist(epi_err, bins=50, color='tomato', alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='green', linestyle='--', label='1px threshold')
    ax.axvline(x=3.0, color='orange', linestyle='--', label='3px threshold')
    ax.set_xlabel("Epipolar Error (px)")
    ax.set_ylabel("Count")
    ax.set_title(f"Epipolar Error Distribution (μ={epi_err.mean():.2f}px)", fontsize=12)
    ax.legend()

    # (d) Model disparity histogram
    ax = axes[1, 0]
    ax.hist(disp_model_v, bins=50, color='steelblue', alpha=0.7, edgecolor='black', label='PI-DINOv3')
    if ncc_valid.sum() > 0:
        ax.hist(ncc_v, bins=50, color='orange', alpha=0.6, edgecolor='black', label='NCC')
    ax.set_xlabel("Disparity (px)")
    ax.set_ylabel("Count")
    ax.set_title("Disparity Distribution: Model vs NCC", fontsize=12)
    ax.legend()

    # (e) Scatter: Model vs NCC disparity (for points both matched)
    ax = axes[1, 1]
    if ncc_valid.sum() > 10:
        ax.scatter(ncc_disps[ncc_valid], disp_model_v[ncc_valid], s=5, alpha=0.5, c='purple')
        lims = [0, max(ncc_disps[ncc_valid].max(), disp_model_v[ncc_valid].max()) * 1.1]
        ax.plot(lims, lims, 'r--', label='y=x (perfect agreement)')
        ax.set_xlabel("NCC Disparity (px)")
        ax.set_ylabel("Model Disparity (px)")
        ax.set_title("Model vs NCC Disparity", fontsize=12)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough NCC matches", ha='center', va='center')

    # (f) 关键点置信度分布 (blob 检测器返回的 scores)
    ax = axes[1, 2]
    kp_scores = out['scores_left'][0].cpu().float().numpy()
    valid_scores = kp_scores[valid]
    ax.hist(valid_scores, bins=50, color='teal', alpha=0.7, edgecolor='black')
    ax.set_xlabel("Keypoint Score (blob size)")
    ax.set_ylabel("Count")
    ax.set_title(f"Keypoint Confidence (μ={valid_scores.mean():.3f})", fontsize=12)
    ax.legend()

    plt.suptitle(f"Model Diagnostic: {bn}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"\n诊断图已保存: {args.output}")


if __name__ == "__main__":
    main()
