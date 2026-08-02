# DINOv3/compute_desc_tracks.py
"""
在矫正灰度图上计算 2D 轨迹点处的 DINOv3 描述子（供 rematch_dino_assisted 的
v2 轨迹辅助匹配）。轨迹点是矫正系坐标（检测在矫正图上做），故先在原始图上
按标定 remap 矫正，再逐帧前向后按轨迹点双线性采样，不存稠密特征图。
（2026-07-29 修复：旧版直接在原始图采样，坐标错位 10-24px。）

用法：../.venv_fs/Scripts/python.exe compute_desc_tracks.py <left|right> [轨迹pkl 输出pkl]
输出：desc_v2tracks_{side}.pkl —— list[dict{frame: (768,) fp32}]，与轨迹顺序一致
"""

import os
import pickle
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_descriptor_nn import sample_desc  # noqa: E402
from utils import pad_to_patch_size  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH = 16


def main():
    side = sys.argv[1]
    img_dir = os.path.join(ROOT, "data", f"{side}_images")
    traj_pkl = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, "data/trajectories", f"trajectories_2d_{side}_jumpcut.pkl")  # 跳切清洗后的生产输入
    out_pkl = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, f"desc_v2tracks_{side}.pkl")

    # v2 轨迹 pkl 是 __main__.Track 系列，用 rematch 的桩类接管
    sys.path.insert(0, os.path.join(ROOT, "particle_processing"))
    import rematch_rectified as rr
    import __main__
    for _n in ["Track", "UltraTrack", "WaveParticleTrack", "StrictTrack",
               "SimpleKalmanFilter", "ImprovedKalmanFilter",
               "ExtendedKalmanFilter", "OptimizedExtendedKalmanFilter",
               "UltraOptimizedKalmanFilter", "WaveParticleKalmanFilter",
               "StrictWaveKalmanFilter"]:
        setattr(__main__, _n, getattr(rr, _n))
    # rematch_rectified 无 RobustKalmanFilter（pkl 里它是 __main__ 下的
    # SimpleKalmanFilter 子类），与 rematch_dino_v2.py 同样方式打桩
    __main__.RobustKalmanFilter = type(
        "RobustKalmanFilter", (rr.SimpleKalmanFilter,), {})
    tracks = pickle.load(open(traj_pkl, "rb"))
    print(f"轨迹 {len(tracks)} 条")

    # 每帧有哪些轨迹点
    per_frame = {}
    for ti, t in enumerate(tracks):
        for f, pt in t.points.items():
            per_frame.setdefault(f, []).append((ti, pt))

    from transformers import AutoModel
    dino = AutoModel.from_pretrained(os.path.join(HERE, "dinov3-base-model"),
                                     local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False

    exts = (".bmp", ".png", ".jpg")
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))
    # 轨迹点是矫正系坐标（检测在矫正图上做），描述子必须在矫正图上采样；
    # 旧版直接在原始图采样，坐标错位 10-24px（2026-07-29 修复）
    calib = np.load(os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz"))
    map1, map2 = calib[f"map1_{side}"], calib[f"map2_{side}"]
    out = [dict() for _ in tracks]
    t0 = __import__("time").time()
    for fi, fname in enumerate(img_files):
        if fi not in per_frame:
            continue
        img = cv2.imread(os.path.join(img_dir, fname), 0)
        if img is None:
            continue
        img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        t = pad_to_patch_size(t, patch_size=PATCH)[0]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(device.type == "cuda")):
            hs = dino(t.to(device)).last_hidden_state
        B, _, H, W = t.shape
        nh, nw = H // PATCH, W // PATCH
        feat = hs[:, -(nh * nw):].transpose(1, 2).reshape(1, -1, nh, nw)[0].cpu()
        items = per_frame[fi]
        kps = torch.tensor(np.array([pt for _, pt in items]), dtype=torch.float32)
        descs = sample_desc(feat, kps).numpy()
        for (ti, _), d in zip(items, descs):
            out[ti][fi] = d.astype(np.float32)
        if (fi + 1) % 200 == 0:
            print(f"{fi + 1}/{len(img_files)}  {(__import__('time').time() - t0):.0f}s")

    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    cov = np.mean([len(d) for d in out])
    print(f"[输出] {out_pkl} | 平均每轨迹描述子数 {cov:.0f}")


if __name__ == "__main__":
    main()
