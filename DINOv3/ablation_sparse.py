"""
DINOv3 Wave Reconstruction - Sparse Ablation Study (with Baseline)
===================================================================
Compares PI-DINOv3 against NCC template matching baseline under
progressively sparser keypoint observations.

Metrics:
  - Epipolar error (y-coordinate alignment)
  - Disparity stability
  - Valid keypoint count

Usage:
    python ablation_sparse.py --checkpoint path/to/best_model.pth
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models import CorrMatchingStereoModel, SparseKeypointDetector
from dataset import RectifiedWaveStereoDataset, stereo_collate_fn
from utils import pad_to_patch_size, downsample_keypoints, load_model_checkpoint



# ============================================================
# NCC Baseline: Template matching along epipolar line
# ============================================================

def ncc_match_keypoints(left_gray_np, right_gray_np, keypoints_left, 
                        patch_size=21, search_range=1000):
    """
    For each left keypoint, search along the epipolar line (same y) in the
    right image using NCC template matching.
    
    Returns:
        disparities: [N] disparity for each keypoint
        epi_errors: [N] epipolar error (always 0 for NCC since we search on same row)
        valid_mask: [N] bool, whether a valid match was found
    """
    half = patch_size // 2
    H, W = left_gray_np.shape
    N = len(keypoints_left)
    
    disparities = np.zeros(N)
    valid_mask = np.zeros(N, dtype=bool)
    
    for i in range(N):
        x_l, y_l = int(keypoints_left[i, 0]), int(keypoints_left[i, 1])
        
        # Bounds check
        if (y_l - half < 0 or y_l + half >= H or 
            x_l - half < 0 or x_l + half >= W):
            continue
            
        # Left template (patch)
        patch_l = left_gray_np[y_l - half:y_l + half + 1, 
                               x_l - half:x_l + half + 1].astype(np.float32)
        
        if patch_l.std() < 1.0:
            continue
            
        # Defines Epipolar line search region in the right image
        # Same y, search x extending to the left (expected positive disparity up to search_range)
        x_start = max(half, x_l - search_range)
        x_end = max(half, x_l - 1)
        
        if x_start >= x_end:
            continue
            
        # Extract the horizontal strip from right image
        search_region = right_gray_np[y_l - half:y_l + half + 1, 
                                      x_start - half : x_end + half + 1].astype(np.float32)
        
        if search_region.shape[1] < patch_l.shape[1]:
            continue
            
        # Fast NCC using OpenCV native operation
        res = cv2.matchTemplate(search_region, patch_l, cv2.TM_CCOEFF_NORMED)
        
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.3:
            # max_loc[0] is the relative x offset within search_region
            # global_x = x_start - half + max_loc[0] + half = x_start + max_loc[0]
            best_x = x_start + max_loc[0]
            disparities[i] = x_l - best_x
            valid_mask[i] = True
            
    return disparities, valid_mask


# ============================================================
# Run ablation for one method at one ratio
# ============================================================

def run_model_ratio(model, cfg, val_loader, device, ratio):
    """Run PI-DINOv3 at given keep ratio."""
    cfg.KEEP_RATIO = ratio
    
    epi_errors = []
    disp_values = []
    num_keypoints = []

    for batch in tqdm(val_loader, desc=f"DINOv3 {ratio:.0%}", leave=False):
        if batch is None:
            continue
        lg = batch['left_gray'].to(device)
        rg = batch['right_gray'].to(device)
        mask = batch['mask'].to(device)
        # 缓存模式下没有 left_rgb/right_rgb，用灰度图复制 3 通道代替
        if batch.get('cached', False):
            lrgb = lg.repeat(1, 3, 1, 1)
            rrgb = rg.repeat(1, 3, 1, 1)
        else:
            lrgb = batch['left_rgb'].to(device)
            rrgb = batch['right_rgb'].to(device)
        patch_size = model.ext.patch
        lg, rg, lrgb, rrgb, mask = pad_to_patch_size(lg, rg, lrgb, rrgb, mask, patch_size=patch_size)

        out = model(lg, rg, lrgb, rrgb, mask)

        kpl = out['keypoints_left'][0]
        kpr_pred = out['keypoints_right_pred'][0]
        scores = out['scores_left'][0]
        disp = out['disparity'][0]

        valid = scores > 0
        n_valid = valid.sum().item()
        if n_valid == 0:
            continue

        epi_err = (kpl[valid, 1] - kpr_pred[valid, 1]).abs().mean().item()
        mean_disp = disp[valid].abs().mean().item()

        epi_errors.append(epi_err)
        disp_values.append(mean_disp)
        num_keypoints.append(n_valid)

    cfg.KEEP_RATIO = 1.0
    return {
        'epi_mean': np.mean(epi_errors) if epi_errors else 0,
        'epi_std': np.std(epi_errors) if epi_errors else 0,
        'disp_mean': np.mean(disp_values) if disp_values else 0,
        'disp_std': np.std(disp_values) if disp_values else 0,
        'avg_kp': np.mean(num_keypoints) if num_keypoints else 0,
    }


def run_ncc_ratio(detector, val_loader, device, ratio):
    """Run NCC baseline at given keep ratio."""
    epi_errors = []
    disp_values = []
    num_keypoints = []

    for batch in tqdm(val_loader, desc=f"NCC    {ratio:.0%}", leave=False):
        if batch is None:
            continue
        lg = batch['left_gray'].to(device)
        rg = batch['right_gray'].to(device)
        mask = batch['mask'].to(device)

        # Detect keypoints
        kpl, sl = detector(lg, mask)
        
        # Downsample
        kpl, sl = downsample_keypoints(kpl, sl, ratio)
        
        # Get numpy images
        l_np = (lg[0, 0].cpu().numpy() * 255).astype(np.uint8)
        r_np = (rg[0, 0].cpu().numpy() * 255).astype(np.uint8)
        kp_np = kpl[0].cpu().numpy()
        sc_np = sl[0].cpu().numpy()
        
        valid_kp = sc_np > 0
        kp_valid = kp_np[valid_kp]
        
        if len(kp_valid) < 3:
            continue
        
        # NCC matching
        disps, match_valid = ncc_match_keypoints(l_np, r_np, kp_valid)
        
        n_matched = match_valid.sum()
        if n_matched < 1:
            continue
        
        # NCC always searches on same y → epipolar error = 0 by design
        # So we measure match quality via disparity consistency (std)
        valid_disps = disps[match_valid]
        epi_errors.append(0.0)  # By construction
        disp_values.append(np.mean(np.abs(valid_disps)))
        num_keypoints.append(n_matched)

    return {
        'epi_mean': 0.0,  # NCC searches on epipolar line
        'epi_std': 0.0,
        'disp_mean': np.mean(disp_values) if disp_values else 0,
        'disp_std': np.std(disp_values) if disp_values else 0,
        'avg_kp': np.mean(num_keypoints) if num_keypoints else 0,
    }


# ============================================================
# Plotting
# ============================================================

def plot_results(ratios, model_results, ncc_results, output_path):
    """Generate publication-quality comparison plot."""
    ratio_pct = [r * 100 for r in ratios]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Valid Match Rate (keypoints that produced valid matches)
    ax1 = axes[0]
    model_kp = [r['avg_kp'] for r in model_results]
    ncc_kp = [r['avg_kp'] for r in ncc_results]
    # Normalize to percentage of 100% baseline
    model_kp_pct = [k / model_kp[0] * 100 if model_kp[0] > 0 else 0 for k in model_kp]
    ncc_kp_pct = [k / ncc_kp[0] * 100 if ncc_kp[0] > 0 else 0 for k in ncc_kp]
    ax1.plot(ratio_pct, model_kp_pct, 'ro-', linewidth=2, markersize=8, label="PI-DINOv3 (Ours)")
    ax1.plot(ratio_pct, ncc_kp_pct, 'b^--', linewidth=2, markersize=8, label="NCC Baseline")
    ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.4)
    ax1.invert_xaxis()
    ax1.set_xlabel("Keypoint Keep Ratio (%)", fontsize=12)
    ax1.set_ylabel("Valid Match Rate (%)", fontsize=12)
    ax1.set_title("(a) Match Success Rate", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=10)

    # 2. Disparity Stability (relative to 100% baseline)
    ax2 = axes[1]
    model_disp = [r['disp_mean'] for r in model_results]
    ncc_disp = [r['disp_mean'] for r in ncc_results]
    base_m = model_disp[0] if model_disp[0] > 0 else 1
    base_n = ncc_disp[0] if ncc_disp[0] > 0 else 1
    model_disp_rel = [abs(d - base_m) / base_m * 100 for d in model_disp]
    ncc_disp_rel = [abs(d - base_n) / base_n * 100 for d in ncc_disp]
    ax2.plot(ratio_pct, model_disp_rel, 'ro-', linewidth=2, markersize=8, label="PI-DINOv3 (Ours)")
    ax2.plot(ratio_pct, ncc_disp_rel, 'b^--', linewidth=2, markersize=8, label="NCC Baseline")
    ax2.invert_xaxis()
    ax2.set_xlabel("Keypoint Keep Ratio (%)", fontsize=12)
    ax2.set_ylabel("Disparity Deviation (%)", fontsize=12)
    ax2.set_title("(b) Disparity Stability", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=10)

    # 3. Disparity Variance (consistency under sparsity)
    ax3 = axes[2]
    model_dstd = [r['disp_std'] for r in model_results]
    ncc_dstd = [r['disp_std'] for r in ncc_results]
    ax3.plot(ratio_pct, model_dstd, 'ro-', linewidth=2, markersize=8, label="PI-DINOv3 (Ours)")
    ax3.plot(ratio_pct, ncc_dstd, 'b^--', linewidth=2, markersize=8, label="NCC Baseline")
    ax3.invert_xaxis()
    ax3.set_xlabel("Keypoint Keep Ratio (%)", fontsize=12)
    ax3.set_ylabel("Disparity Std. Dev.", fontsize=12)
    ax3.set_title("(c) Prediction Consistency", fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(fontsize=10)

    plt.suptitle("Robustness Under Sparse Observational Data", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DINOv3 Sparse Ablation Study (with Baseline)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="sparse_ablation_curve.png")
    parser.add_argument("--ratios", type=str, default="1.0,0.8,0.5,0.3,0.1,0.05")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Build model
    model = CorrMatchingStereoModel(cfg).to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    load_model_checkpoint(model, args.checkpoint, device, strict=False)
    model.eval()

    # Baseline detector (same blob detector, but no Transformer)
    detector = SparseKeypointDetector(cfg).to(device)

    # Dataset
    val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=stereo_collate_fn)

    keep_ratios = [float(r) for r in args.ratios.split(',')]

    print(f"\n{'='*60}")
    print(f"  Sparse Ablation Study: PI-DINOv3 vs NCC Baseline")
    print(f"  Ratios: {keep_ratios}")
    print(f"  Val frames: {len(val_ds)}")
    print(f"{'='*60}\n")

    model_results = []
    ncc_results = []

    with torch.no_grad():
        for ratio in keep_ratios:
            print(f"\n--- Keep Ratio: {ratio:.0%} ---")
            
            # PI-DINOv3
            m_res = run_model_ratio(model, cfg, val_loader, device, ratio)
            model_results.append(m_res)
            print(f"  [PI-DINOv3] Disp: {m_res['disp_mean']:.1f}±{m_res['disp_std']:.1f} | "
                  f"EpiErr: {m_res['epi_mean']:.3f} | KPs: {m_res['avg_kp']:.0f}")

            # NCC Baseline
            n_res = run_ncc_ratio(detector, val_loader, device, ratio)
            ncc_results.append(n_res)
            print(f"  [NCC      ] Disp: {n_res['disp_mean']:.1f}±{n_res['disp_std']:.1f} | "
                  f"KPs: {n_res['avg_kp']:.0f}")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Ratio':>8} | {'--- PI-DINOv3 ---':^30} | {'--- NCC Baseline ---':^30}")
    print(f"{'':>8} | {'Disp':>10} {'EpiErr':>8} {'KPs':>6} | {'Disp':>10} {'':>8} {'KPs':>6}")
    print(f"{'-'*80}")
    for i, r in enumerate(keep_ratios):
        m = model_results[i]
        n = ncc_results[i]
        print(f"{r:>7.0%} | {m['disp_mean']:>8.1f}±{m['disp_std']:<4.0f} "
              f"{m['epi_mean']:>7.2f}  {m['avg_kp']:>5.0f} | "
              f"{n['disp_mean']:>8.1f}±{n['disp_std']:<4.0f} "
              f"{'—':>7}  {n['avg_kp']:>5.0f}")
    print(f"{'='*80}")

    # Plot
    plot_results(keep_ratios, model_results, ncc_results, args.output)


if __name__ == "__main__":
    main()
