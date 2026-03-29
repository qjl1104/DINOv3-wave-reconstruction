# DINOv3 Archive

Historical versions of the DINOv3 wave reconstruction project, organized by evolution phase.

## Phase 1: Dense Disparity (Sept 2025)
DINOv2/v3 + 3D Hourglass cost volume approach. Abandoned in favor of sparse matching.

## Phase 2: Multi-AI Exploration (Sept 17-30, 2025)
Experimental scripts generated with Gemini, GPT, Claude, Grok, DeepSeek. 
Key insight: Attention-based matching (0929) became the foundation for Phase 3.

## Phase 3: Sparse Matching Iterations (Oct 9 - Nov 18, 2025)
16 iterations of sparse keypoint + DINO feature + Transformer matching.
Final version: `sparse_reconstructor_1118_gemini.py` (V5 with photometric loss).

## Phase 4: PINN Training (Nov 19-25, 2025)
Physics-Informed Neural Network training with smoothness, slope, and zero-mean constraints.
Key scripts: `train_v22_pinn.py`, `Corrected Inference.py` (V21 with Z-axis fix).

## Winner: V24.10 (Nov 28 - Dec 11, 2025)
The final best version lives in `DINOv3/train.py` and `DINOv3/inference.py`.
It combines: new Float32 calibration + two-phase training + balanced PINN loss.
