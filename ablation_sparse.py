"""
DINOv3 Wave Reconstruction - Sparse Ablation Study
====================================================
Evaluates model robustness under progressively sparser keypoint observations.

For each KEEP_RATIO, computes:
  - Mean epipolar error (y-coordinate alignment)
  - Mean disparity magnitude
  - Number of valid keypoints

The 100% result serves as pseudo ground truth reference.

Usage:
    python ablation_sparse.py --checkpoint path/to/best_model.pth
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models import SparseMatchingStereoModel
from dataset import RectifiedWaveStereoDataset, stereo_collate_fn


def pad_to_14(*tensors):
    """Pad tensors to be divisible by 14."""
    h, w = tensors[0].shape[2:]
    padh = (14 - h % 14) % 14
    padw = (14 - w % 14) % 14
    if padh > 0 or padw > 0:
        return [F.pad(t, (0, padw, 0, padh)) for t in tensors]
    return list(tensors)


def run_single_ratio(model, cfg, val_loader, device, ratio):
    """Run inference at a given KEEP_RATIO and collect metrics."""
    cfg.KEEP_RATIO = ratio
    
    epi_errors = []
    disp_values = []
    num_keypoints = []

    for batch in tqdm(val_loader, desc=f"Ratio {ratio:.0%}", leave=False):
        if batch is None:
            continue

        lg = batch['left_gray'].to(device)
        rg = batch['right_gray'].to(device)
        lrgb = batch['left_rgb'].to(device)
        rrgb = batch['right_rgb'].to(device)
        mask = batch['mask'].to(device)

        lg, rg, lrgb, rrgb, mask = pad_to_14(lg, rg, lrgb, rrgb, mask)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            out = model(lg, rg, lrgb, rrgb, mask, apply_epipolar_mask=True)

        kpl = out['keypoints_left'][0]
        kpr_pred = out['keypoints_right_pred'][0]
        scores = out['scores_left'][0]
        disp = out['disparity'][0]

        valid = scores > 0
        n_valid = valid.sum().item()
        if n_valid == 0:
            continue

        # Epipolar error: y-coordinate difference between predicted matches
        epi_err = (kpl[valid, 1] - kpr_pred[valid, 1]).abs().mean().item()
        
        # Mean disparity (as proxy for reconstruction depth consistency)
        mean_disp = disp[valid].abs().mean().item()

        epi_errors.append(epi_err)
        disp_values.append(mean_disp)
        num_keypoints.append(n_valid)

    return {
        'ratio': ratio,
        'epi_error_mean': np.mean(epi_errors) if epi_errors else 0,
        'epi_error_std': np.std(epi_errors) if epi_errors else 0,
        'disp_mean': np.mean(disp_values) if disp_values else 0,
        'disp_std': np.std(disp_values) if disp_values else 0,
        'avg_keypoints': np.mean(num_keypoints) if num_keypoints else 0,
        'n_frames': len(epi_errors),
    }


def plot_results(results, output_path):
    """Generate publication-quality ablation curves."""
    ratios = [r['ratio'] * 100 for r in results]
    epi_means = [r['epi_error_mean'] for r in results]
    epi_stds = [r['epi_error_std'] for r in results]
    disp_means = [r['disp_mean'] for r in results]
    avg_kps = [r['avg_keypoints'] for r in results]

    # Normalize disparity relative to 100% baseline
    baseline_disp = disp_means[0] if disp_means[0] > 0 else 1
    disp_relative = [d / baseline_disp * 100 for d in disp_means]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Epipolar Error
    ax1 = axes[0]
    ax1.errorbar(ratios, epi_means, yerr=epi_stds, fmt='ro-', linewidth=2,
                 capsize=4, markersize=8, label="PI-DINOv3")
    ax1.invert_xaxis()
    ax1.set_xlabel("Keypoint Keep Ratio (%)", fontsize=12)
    ax1.set_ylabel("Epipolar Error (pixels)", fontsize=12)
    ax1.set_title("(a) Geometric Consistency", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=11)

    # 2. Disparity Stability
    ax2 = axes[1]
    ax2.plot(ratios, disp_relative, 'bs-', linewidth=2, markersize=8, label="Relative Disparity")
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label="100% Baseline")
    ax2.invert_xaxis()
    ax2.set_xlabel("Keypoint Keep Ratio (%)", fontsize=12)
    ax2.set_ylabel("Relative Disparity (%)", fontsize=12)
    ax2.set_title("(b) Disparity Stability", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=11)

    # 3. Effective Keypoints
    ax3 = axes[2]
    ax3.bar(ratios, avg_kps, width=8, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax3.invert_xaxis()
    ax3.set_xlabel("Keypoint Keep Ratio (%)", fontsize=12)
    ax3.set_ylabel("Avg. Valid Keypoints", fontsize=12)
    ax3.set_title("(c) Observation Density", fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.suptitle("Robustness Under Sparse Observational Data", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Sparse Ablation Study")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="sparse_ablation_curve.png",
                        help="Output plot path")
    parser.add_argument("--ratios", type=str, default="1.0,0.8,0.5,0.3,0.1,0.05",
                        help="Comma-separated keep ratios")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Build model
    model = SparseMatchingStereoModel(cfg).to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Validation set
    val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=stereo_collate_fn)

    # Run ablation
    keep_ratios = [float(r) for r in args.ratios.split(',')]
    results = []

    print(f"\n{'='*50}")
    print(f"Sparse Ablation Study")
    print(f"Ratios: {keep_ratios}")
    print(f"Val frames: {len(val_ds)}")
    print(f"{'='*50}\n")

    with torch.no_grad():
        for ratio in keep_ratios:
            result = run_single_ratio(model, cfg, val_loader, device, ratio)
            results.append(result)
            print(f"  Ratio {ratio:>5.0%} | "
                  f"EpiErr: {result['epi_error_mean']:.3f}±{result['epi_error_std']:.3f} px | "
                  f"Disp: {result['disp_mean']:.1f} | "
                  f"KPs: {result['avg_keypoints']:.0f}")

    # Reset ratio
    cfg.KEEP_RATIO = 1.0

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Ratio':>8} | {'Epi Error':>12} | {'Disp Mean':>10} | {'Keypoints':>10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['ratio']:>7.0%} | "
              f"{r['epi_error_mean']:>8.3f}±{r['epi_error_std']:.3f} | "
              f"{r['disp_mean']:>10.1f} | "
              f"{r['avg_keypoints']:>10.0f}")
    print(f"{'='*70}")

    # Plot
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
