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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models import CorrMatchingStereoModel, SparseKeypointDetector
from dataset import RectifiedWaveStereoDataset, stereo_collate_fn
from utils import pad_to_patch_size, downsample_keypoints, load_model_checkpoint



def ncc_match_keypoints_2d(left_gray_np, right_gray_np, keypoints_left, 
                           patch_size=21, search_range_x=1000, search_y=25):
    """
    2D template matching for NCC to expose actual Y-drift (Aperture problem)
    """
    half = patch_size // 2
    H, W = left_gray_np.shape
    N = len(keypoints_left)
    
    disparities = np.zeros(N)
    epi_errors = np.zeros(N)
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
            
        x_start = max(half, x_l - search_range_x)
        x_end = max(half, x_l - 1)
        # 2D search window
        y_start = max(half, y_l - search_y)
        y_end = min(H - half - 1, y_l + search_y)
        
        if x_start >= x_end or y_start >= y_end:
            continue
            
        search_region = right_gray_np[y_start - half : y_end + half + 1, 
                                      x_start - half : x_end + half + 1].astype(np.float32)
        
        if search_region.shape[1] < patch_l.shape[1] or search_region.shape[0] < patch_l.shape[0]:
            continue
            
        res = cv2.matchTemplate(search_region, patch_l, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.3:
            best_x = x_start + max_loc[0]
            best_y = y_start + max_loc[1]
            disparities[i] = x_l - best_x
            epi_errors[i] = abs(y_l - best_y)
            valid_mask[i] = True
            
    return disparities, epi_errors, valid_mask


def run_model_ratio(model, cfg, val_loader, device, ratio):
    cfg.KEEP_RATIO = ratio
    all_disp, all_epi, num_kp = [], [], []

    for batch in tqdm(val_loader, desc=f"DINOv3 {ratio:.0%}", leave=False):
        if batch is None: continue
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
        if valid.sum() == 0: continue
        
        disp_vals = disp[valid].abs().cpu().numpy()
        epi_vals = (kpl[valid, 1] - kpr_pred[valid, 1]).abs().cpu().numpy()
        
        all_disp.extend(disp_vals.tolist())
        all_epi.extend(epi_vals.tolist())
        num_kp.append(valid.sum().item())

    cfg.KEEP_RATIO = 1.0
    all_disp = np.array(all_disp)
    all_epi = np.array(all_epi)
    
    # True wave disparity is approx 610-620. Aliasing trap is typically ~550px.
    # Outliers are points that fall into the false wave period.
    outliers = np.sum((all_disp < 580) | (all_disp > 650))
    
    return {
        'disp_mean': np.mean(all_disp) if len(all_disp) else 0,
        'epi_mean': np.mean(all_epi) if len(all_epi) else 0,
        'outlier_rate': outliers / len(all_disp) if len(all_disp) else 0,
        'avg_kp': np.mean(num_kp) if num_kp else 0,
        'raw_disp': all_disp
    }


def run_ncc_ratio(detector, val_loader, device, ratio):
    all_disp, all_epi, num_kp = [], [], []

    for batch in tqdm(val_loader, desc=f"NCC    {ratio:.0%}", leave=False):
        if batch is None: continue
        lg = batch['left_gray'].to(device)
        rg = batch['right_gray'].to(device)
        mask = batch['mask'].to(device)

        kpl, sl = detector(lg, mask)
        kpl, sl = downsample_keypoints(kpl, sl, ratio)
        
        l_np = (lg[0, 0].cpu().numpy() * 255).astype(np.uint8)
        r_np = (rg[0, 0].cpu().numpy() * 255).astype(np.uint8)
        kp_np = kpl[0].cpu().numpy()
        sc_np = sl[0].cpu().numpy()
        
        valid_kp = sc_np > 0
        kp_valid = kp_np[valid_kp]
        if len(kp_valid) < 3: continue
        
        disps, epi_errs, match_valid = ncc_match_keypoints_2d(l_np, r_np, kp_valid)
        n_matched = match_valid.sum()
        if n_matched < 1: continue
        
        valid_disps = disps[match_valid]
        valid_epis = epi_errs[match_valid]
        
        all_disp.extend(valid_disps.tolist())
        all_epi.extend(valid_epis.tolist())
        num_kp.append(n_matched)

    all_disp = np.array(all_disp)
    all_epi = np.array(all_epi)
    outliers = np.sum((all_disp < 580) | (all_disp > 650))
    
    return {
        'disp_mean': np.mean(all_disp) if len(all_disp) else 0,
        'epi_mean': np.mean(all_epi) if len(all_epi) else 0,
        'outlier_rate': outliers / len(all_disp) if len(all_disp) else 0,
        'avg_kp': np.mean(num_kp) if num_kp else 0,
        'raw_disp': all_disp
    }


def generate_paper_figures(keep_ratios, m_res, n_res, dir_out="."):
    print("\nGenerating Paper Figures...")
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # --- Plot 1: Histogram at 100% ---
    m_raw = m_res[0]['raw_disp']
    n_raw = n_res[0]['raw_disp']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(n_raw, bins=100, range=(450, 750), alpha=0.6, color='#1f77b4', edgecolor='black', density=False, label='NCC (Captured by Aliasing Local Optima)')
    ax.hist(m_raw, bins=100, range=(450, 750), alpha=0.7, color='#d62728', edgecolor='black', density=False, label='PI-DINOv3 (Bypasses Textural Aliasing)')
    ax.axvspan(580, 650, color='gray', alpha=0.15, label='Physical Solution Space')
    
    ax.set_title("Stereo Matching Disparity Distribution on Water Waves", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Disparity (px) [Translates to Depth]", fontsize=12)
    ax.set_ylabel("Number of Keypoints", fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    path1 = os.path.join(dir_out, "paper_fig_distribution.png")
    plt.tight_layout()
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Cyclic Aliasing & Epipolar Drift ---
    ratio_pct = [r * 100 for r in keep_ratios]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (A) Alias Error
    ax1 = axes[0]
    m_out = [r['outlier_rate'] * 100 for r in m_res]
    n_out = [r['outlier_rate'] * 100 for r in n_res]
    
    ax1.plot(ratio_pct, m_out, 'ro-', linewidth=2.5, markersize=8, label='PI-DINOv3 (Global Prior)')
    ax1.plot(ratio_pct, n_out, 'b^--', linewidth=2.5, markersize=8, label='NCC Baseline (2D Template)')
    ax1.invert_xaxis()
    ax1.set_xlabel("Keypoint Observation Ratio (%)", fontsize=12)
    ax1.set_ylabel("Mismatch / Aliasing Rate (%)", fontsize=12)
    ax1.set_title("(a) Cyclic Aliasing Vulnerability", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # (B) Epipolar Drift
    ax2 = axes[1]
    m_epi = [r['epi_mean'] for r in m_res]
    n_epi = [r['epi_mean'] for r in n_res]
    
    ax2.plot(ratio_pct, m_epi, 'ro-', linewidth=2.5, markersize=8, label='PI-DINOv3')
    ax2.plot(ratio_pct, n_epi, 'b^--', linewidth=2.5, markersize=8, label='NCC Baseline (2D Drift)')
    ax2.invert_xaxis()
    ax2.set_xlabel("Keypoint Observation Ratio (%)", fontsize=12)
    ax2.set_ylabel("Mean Vertical Drift / Epipolar Error (px)", fontsize=12)
    ax2.set_title("(b) 2D Vertical Drift Evaluation", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)

    str_title = "Evaluation of Sub-pixel Accuracy & Robustness to Periodic Textures"
    plt.suptitle(str_title, fontsize=16, fontweight='bold', y=1.05)
    
    path2 = os.path.join(dir_out, "paper_fig_robustness.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Done] Exported {path1}")
    print(f"[Done] Exported {path2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ratios", type=str, default="1.0,0.8,0.5,0.3,0.1,0.05")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    model = CorrMatchingStereoModel(cfg).to(device)
    model.eval()
    print(f"Loading checkpoint: {args.checkpoint}")
    load_model_checkpoint(model, args.checkpoint, device, strict=False)

    detector = SparseKeypointDetector(cfg).to(device)

    # Ensure validation subset is evaluated 
    val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=stereo_collate_fn)

    keep_ratios = [float(r) for r in args.ratios.split(',')]

    print(f"\n============================================================")
    print(f"  Generating Superiority Figures for Academic Paper")
    print(f"  Ratios: {keep_ratios}")
    print(f"============================================================\n")

    m_results, n_results = [], []

    with torch.no_grad():
        for i, ratio in enumerate(keep_ratios):
            print(f"Processing R={ratio:.0%} ...")
            m_res = run_model_ratio(model, cfg, val_loader, device, ratio)
            n_res = run_ncc_ratio(detector, val_loader, device, ratio)
            
            m_results.append(m_res)
            n_results.append(n_res)

    generate_paper_figures(keep_ratios, m_results, n_results, dir_out=os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
