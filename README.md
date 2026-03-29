# DINOv3 Wave Reconstruction

Stereo wave surface reconstruction using DINOv2 features + Transformer matching + PINN physics constraints.

## Architecture

```
Left Image  → Blob Keypoints → DINOv2 Features → Transformer Encoder → Soft Matching → 3D Reprojection
Right Image → Blob Keypoints → DINOv2 Features → Transformer Encoder ↗
```

The model learns to match sparse keypoints between rectified stereo image pairs, with training guided by:
- **Photometric loss** — matched patches should look similar
- **Epipolar loss** — matched points should satisfy epipolar geometry
- **PINN constraints** — reconstructed surface should be physically plausible (smooth, bounded slope, zero-mean)

## Project Structure

```
DINOv3/
├── config.py              # Unified configuration (all parameters in one place)
├── models.py              # Model definitions (shared across train/inference)
├── losses.py              # PINN physics loss functions
├── dataset.py             # Stereo dataset with rectification
├── train.py               # Training script (CosineAnnealingLR, validation, best model saving)
├── inference.py           # Single-frame 3D reconstruction
├── temporal_inference.py  # Multi-frame temporal wave analysis
├── generate_calibration.py # Camera calibration parameter generation
└── _archive/              # Historical versions and development notes
```

## Requirements

- Python 3.10+
- PyTorch 2.6+ (with CUDA)
- transformers (HuggingFace, for DINOv2)
- OpenCV
- scipy, scikit-learn, pandas, matplotlib

## Usage

### Training
```bash
conda activate dino
cd DINOv3
python train.py
```

Adjust parameters in `config.py` (learning rate, loss weights, etc.).
To fine-tune from a pretrained checkpoint, set `PRETRAINED_CHECKPOINT` in `config.py`.

### Single-Frame Inference
```bash
python inference.py --checkpoint path/to/best_model.pth --image_index 0
```

### Temporal Analysis
```bash
python temporal_inference.py --model_path path/to/best_model.pth --limit 500
```

## Data Setup

Place your data outside this directory:
```
wave_reconstruction_project/
├── data/
│   ├── left_images/    # left*.png
│   └── right_images/   # right*.png
└── DINOv3/             # this repo
```

Calibration file (`paper_params_recalculated.npz`) should be placed in `1128/`.
DINOv2 weights should be in `dinov3-base-model/`.
