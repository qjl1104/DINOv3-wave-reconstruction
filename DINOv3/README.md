# PI-DINOv3: Physics-Informed Sparse Stereo Matching for Laboratory Wave Surface Reconstruction

A stereo vision system that reconstructs 3D water wave surfaces from binocular camera images. The core idea is to combine **DINOv2 vision foundation model** features with a **Transformer-based cross-view matcher**, trained end-to-end using **PINN (Physics-Informed Neural Network) constraints** — enabling robust matching on texturally repetitive water surfaces where traditional methods (e.g., NCC template matching) suffer from cyclic aliasing.

## Method Overview

```
                          ┌──────────────────────────────────────────────────────┐
                          │              PI-DINOv3 Stereo Pipeline               │
                          └──────────────────────────────────────────────────────┘

  Rectified Left Image ──┬──► Blob Detector ──► Left Keypoints (N×2)
                         │                            │
                         │                            ▼
                         └──► DINOv2 Backbone ──► Feature Sampling ──► Left Descriptors (N×768)
                                (frozen)                                       │
                                                                    ┌──────────┘
                                                                    ▼
                                                        Positional Encoding
                                                                    │
                                                                    ▼
                                                       Transformer Encoder (×6)
                                                                    │
                                                                    ▼
  Rectified Right Image ──┬──► Blob Detector ──► Right Keypoints (M×2)        │
                          │                            │                       │
                          │                            ▼                       │
                          └──► DINOv2 Backbone ──► Feature Sampling ──► Right Descriptors (M×768)
                                 (frozen)                                      │
                                                                    ┌──────────┘
                                                                    ▼
                                                        Positional Encoding
                                                                    │
                                                                    ▼
                                                       Transformer Encoder (×6)
                                                                    │
                                                                    ▼
                                                  Cosine Similarity Matrix (N×M)
                                                                    │
                                                         ┌──────────┤
                                                         │          │
                                                  [Training]   [Inference]
                                                  Softmax        Softmax + Epipolar Mask
                                                         │          │
                                                         ▼          ▼
                                                  Soft-Argmax Match → Predicted Right Coords
                                                                    │
                                                                    ▼
                                                        Disparity = x_left - x_right
                                                                    │
                                                                    ▼
                                                     Q-Matrix Reprojection → 3D Points
```

### Training Loss (Self-Supervised + Physics-Informed)

The model is trained **without ground-truth disparity labels**. Supervision comes entirely from geometric and physical priors:

| Loss Component | Weight | Description |
|---|---|---|
| **Photometric** | 5.0 | Patch-based L1 between left keypoint and predicted right keypoint, weighted by brightness |
| **Epipolar** | 0.1 | Penalizes y-coordinate difference between matched points (should be zero for rectified pairs) |
| **Smoothness** (PINN) | 2.0 | KNN-based local height smoothness on the reconstructed 3D surface |
| **Slope** (PINN) | 0.1 | Penalizes physically implausible surface gradients (> 0.4 rad) |
| **Zero-Mean** (PINN) | 0.1 | Encourages the mean wave surface height to be near zero |

## Project Structure

```
DINOv3/
├── config.py                  # Unified configuration (@dataclass with all hyperparameters)
├── models.py                  # Model definitions
│                              #   - SparseKeypointDetector (OpenCV Blob)
│                              #   - DINOv3FeatureExtractor (frozen DINOv2 + grid_sample)
│                              #   - SparseMatchingStereoModel (full pipeline)
├── losses.py                  # PINNPhysicsLoss (photometric + epipolar + physics)
├── dataset.py                 # RectifiedWaveStereoDataset (auto-rectification, 90/10 split)
├── utils.py                   # Shared utilities (pad_to_14, reproject_to_3d, etc.)
├── train.py                   # Training script
│                              #   - AMP mixed precision, gradient accumulation
│                              #   - CosineAnnealingLR, validation every 5 epochs
│                              #   - Best model saving, JSON logging, loss plots
├── inference.py               # Single-frame inference → 3D point cloud + wave fitting
├── temporal_inference.py      # Multi-frame temporal analysis
│                              #   - Phase 1: Global water surface calibration (RANSAC)
│                              #   - Phase 2: Per-frame wave height & wavelength
│                              #   - Phase 3: Statistical & frequency-domain analysis
├── ablation_sparse.py         # Sparsity ablation: PI-DINOv3 vs NCC at varying keep ratios
├── paper_figure_generator.py  # Publication figure generator (aliasing & drift evaluation)
├── diagnose_model.py          # Visual diagnostics (matching lines, disparity histograms)
├── generate_calibration.py    # Stereo calibration file generation from MATLAB params
├── gen_figure.py              # Placeholder RMSE comparison plot
└── _archive/                  # Historical development versions (4 phases)
    ├── phase1_dense/          #   Dense stereo attempts
    ├── phase2_multi_ai/       #   Multi-AI-assistant iterations
    ├── phase3_sparse_iter/    #   Sparse reconstruction iterations
    ├── phase4_pinn_v22/       #   PINN-integrated versions
    └── utilities/             #   Legacy analysis tools
```

## Requirements

- Python 3.10+
- PyTorch 2.6+ (CUDA recommended; CPU fallback supported)
- transformers (HuggingFace — for DINOv2 backbone)
- OpenCV (cv2)
- scipy, scikit-learn, pandas
- matplotlib, tqdm

Install dependencies:

```bash
pip install torch torchvision transformers opencv-python scipy scikit-learn pandas matplotlib tqdm
```

## Data Setup

```
wave_reconstruction_project/
├── data/
│   ├── left_images/           # Stereo left images (e.g., left_00001.bmp)
│   └── right_images/          # Stereo right images (e.g., right_00001.bmp)
├── DINOv3/
│   ├── dinov3-base-model/     # DINOv2 weights (facebook/dinov2-base)
│   ├── 1128/
│   │   └── paper_params_recalculated.npz   # Calibration file
│   └── ...                    # Source code
```

### Calibration File

The calibration `.npz` file must contain:
- `map1_left`, `map2_left`, `map1_right`, `map2_right` — Rectification remap tables (float32)
- `Q` — 4x4 disparity-to-depth reprojection matrix

To generate from MATLAB stereo calibration parameters:

```bash
python generate_calibration.py
```

### DINOv2 Weights

The model auto-downloads `facebook/dinov2-base` from HuggingFace on first run. For offline use, pre-download to `dinov3-base-model/`:

```bash
python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('facebook/dinov2-base'); m.save_pretrained('dinov3-base-model')"
```

## Usage

### Training

```bash
cd DINOv3
python train.py
```

All hyperparameters are configured in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `LEARNING_RATE` | 2e-4 | AdamW learning rate |
| `NUM_EPOCHS` | 150 | Total training epochs |
| `ACCUMULATION_STEPS` | 4 | Gradient accumulation steps (effective batch = 4) |
| `NUM_ATTENTION_LAYERS` | 6 | Transformer encoder depth |
| `MATCHING_TEMPERATURE` | 15.0 | Softmax temperature for similarity scores |
| `PRETRAINED_CHECKPOINT` | "" | Path to resume/fine-tune from (leave empty for scratch) |

Training outputs are saved to `training_runs/<timestamp>/`:

```
training_runs/20260418-143000/
├── checkpoints/
│   ├── best_model.pth       # Best validation loss
│   ├── model_ep10.pth       # Periodic checkpoints (every 10 epochs)
│   └── ...
├── logs/
│   └── training_log.json    # Full loss history
├── vis/
│   └── loss_history.png     # Training curves
└── config.json              # Frozen config snapshot
```

### Single-Frame Inference

Reconstruct 3D wave surface from one stereo pair:

```bash
python inference.py --checkpoint training_runs/.../checkpoints/best_model.pth \
                    --image_index 0 \
                    --output result.png
```

Output: a figure with 3D point cloud, side-view wave profile with cosine fit, and the original image.

### Temporal Analysis

Process a sequence of frames for wave height/wavelength time series:

```bash
python temporal_inference.py --model_path path/to/best_model.pth \
                             --limit 500 \
                             --output_csv wave_data.csv
```

Three-phase pipeline:
1. **Phase 1** — Sample 20 frames to compute a global water surface reference plane (RANSAC plane fitting)
2. **Phase 2** — Process all frames: reproject to 3D, rotate to world coordinates, fit 2D cosine wave
3. **Phase 3** — Output wave height/wavelength distributions, frequency analysis via dispersion relation

### Ablation Study

Compare PI-DINOv3 against NCC baseline under progressively sparser keypoint observations:

```bash
python ablation_sparse.py --checkpoint path/to/best_model.pth \
                          --ratios 1.0,0.8,0.5,0.3,0.1,0.05 \
                          --output sparse_ablation_curve.png
```

Metrics: epipolar error, disparity stability, valid match rate — all plotted as functions of keypoint keep ratio.

### Model Diagnostics

Generate a 6-panel diagnostic figure for a single frame:

```bash
python diagnose_model.py --checkpoint path/to/best_model.pth \
                         --image_index 0 \
                         --output diagnostic.png
```

Panels: (a) keypoint overlay, (b) matching lines with epipolar error color coding, (c) epipolar error histogram, (d) disparity distribution (model vs NCC), (e) model-vs-NCC disparity scatter, (f) match confidence distribution.

### Publication Figures

Generate camera-ready figures for the paper (aliasing analysis + drift evaluation):

```bash
python paper_figure_generator.py --checkpoint path/to/best_model.pth \
                                 --ratios 1.0,0.8,0.5,0.3,0.1,0.05
```

## Key Design Decisions

### Why Sparse (Not Dense) Stereo?

Water surfaces have repetitive sinusoidal textures. Dense stereo methods (SGM, RAFT-Stereo) produce systematic errors due to **cyclic aliasing** — multiple wave peaks produce equally plausible matches. By using sparse keypoints (blob detections on tracer particles or wave crests), the model focuses on distinctive features and uses global Transformer attention to disambiguate.

### Why PINN Losses?

Without ground-truth depth labels, the model could converge to photometrically consistent but physically implausible solutions (e.g., noisy surfaces, extreme slopes). PINN constraints act as regularizers:
- **Smoothness** prevents discontinuous surfaces
- **Slope penalty** enforces physical wave steepness limits
- **Zero-mean** anchors the surface to a stable reference level

### Why DINOv2 (Frozen)?

DINOv2's self-supervised features are semantic and spatially coherent — they distinguish wave crests from troughs even when pixel intensities are similar. Freezing the backbone avoids catastrophic forgetting on the small dataset and reduces trainable parameters to only the Transformer matcher (~18M params).

### Training vs Inference Difference

During training, the epipolar constraint is handled by the loss function (soft penalty). During inference, a hard epipolar mask is applied before softmax: only right keypoints within `EPIPOLAR_THRESHOLD` (3 px) of the left keypoint's y-coordinate are considered as candidates. This ensures geometric consistency at test time.
