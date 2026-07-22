# DINOv3/compute_desc_92.py
"""
DINOv3 检测分类器 第1步：在 preprocessed_92 图像上计算检测点描述子。
输入约定与 precompute_cache.py 一致（灰度→3ch→/255→pad 到 patch 倍数，
bf16，取 last_hidden_state 的 patch tokens）。
只保存检测点处的双线性采样描述子（不存稠密特征图，省 ~48GB 磁盘）。

用法：../.venv_fs/Scripts/python.exe compute_desc_92.py left
      ../.venv_fs/Scripts/python.exe compute_desc_92.py right
输出：desc_92_{side}.pkl —— list[(N,768) fp16]，与 detections_92_{side}.pkl 逐点对齐
"""

import glob
import os
import pickle
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_descriptor_nn import sample_desc  # noqa: E402
from utils import pad_to_patch_size as pad_to_patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH = 16


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "left"
    img_dir = os.path.join(ROOT, "data/preprocessed_92", side)
    det_pkl = os.path.join(ROOT, "data/detections", f"detections_92_{side}.pkl")
    out_pkl = os.path.join(HERE, f"desc_92_{side}.pkl")

    from transformers import AutoModel
    dino = AutoModel.from_pretrained(os.path.join(HERE, "dinov3-base-model"),
                                     local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False

    detections = pickle.load(open(det_pkl, "rb"))
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert len(files) == len(detections), f"{len(files)} imgs vs {len(detections)} det frames"

    out = []
    t0 = __import__("time").time()
    for fi, (fp, dets) in enumerate(zip(files, detections)):
        img = cv2.imread(fp, 0)
        if img is None or len(dets) == 0:
            out.append(np.zeros((0, 768), dtype=np.float16))
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        t = pad_to_patch(t, patch_size=PATCH)[0]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(device.type == "cuda")):
            hs = dino(t.to(device)).last_hidden_state
        B, _, H, W = t.shape
        nh, nw = H // PATCH, W // PATCH
        feat = hs[:, -(nh * nw):].transpose(1, 2).reshape(1, -1, nh, nw)[0].cpu()
        kps = torch.tensor(np.array(dets), dtype=torch.float32)
        desc = sample_desc(feat, kps).numpy().astype(np.float16)
        out.append(desc)
        if (fi + 1) % 200 == 0:
            print(f"{fi + 1}/{len(files)}  {(__import__('time').time() - t0):.0f}s")

    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"[输出] {out_pkl}")


if __name__ == "__main__":
    main()
