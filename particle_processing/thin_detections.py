# particle_processing/thin_detections.py
"""
粒子密度抽稀（开题"极端稀疏工况性能边界"实验的输入生成器）。

从 canonical v3 检测结果（data/detections/detections_{side}_v3.pkl）中按保留
概率 p 随机剔除【非水沫标识物】检测点；水沫点（meta big_comp=1）原样保留
——模拟"减少投放粒子"而非"改变水情"。每帧独立均匀随机剔除，seed 固定
可复现；左右两侧用同一个 seed 但独立随机流（SeedSequence([seed, side])）。

输出结构与 canonical 完全一致（det pkl 保留水沫点、配对的 _meta.pkl 同步
抽稀），因此下游 03b --no-foam 的泡沫过滤行为与生产链逐点一致。

输入（只读，绝不修改）：
  data/detections/detections_{side}_v3.pkl
  data/detections/detections_{side}_v3_meta.pkl
输出：
  data/detections/thin/detections_{side}_v3_thin{pct}_s{seed}.pkl
  data/detections/thin/detections_{side}_v3_thin{pct}_s{seed}_meta.pkl

用法：../.venv_fs/Scripts/python.exe thin_detections.py <p> <seed> [left|right|both]
  例：thin_detections.py 0.5 1        # p=0.5, seed=1, 双侧
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "data", "detections")
OUT_DIR = os.path.join(SRC_DIR, "thin")


def pct_str(p):
    return str(int(round(p * 100)))


def thin_side(side, p, seed):
    det_pkl = os.path.join(SRC_DIR, f"detections_{side}_v3.pkl")
    meta_pkl = det_pkl.replace(".pkl", "_meta.pkl")
    with open(det_pkl, "rb") as f:
        dets = pickle.load(f)
    with open(meta_pkl, "rb") as f:
        metas = pickle.load(f)
    assert len(dets) == len(metas), f"{side}: det/meta 帧数不一致"

    # 同 seed、两侧独立随机流（side 编码进 entropy，左右不共享序列）
    rng = np.random.default_rng([seed, 0 if side == "left" else 1])
    tag = f"thin{pct_str(p)}_s{seed}"
    out_det = os.path.join(OUT_DIR, f"detections_{side}_v3_{tag}.pkl")
    out_meta = out_det.replace(".pkl", "_meta.pkl")

    dets_out, metas_out = [], []
    n_tot = n_foam = n_keep = n_keep_nonfoam = 0
    for fi, (d, mm) in enumerate(zip(dets, metas)):
        mm = np.asarray(mm)
        assert mm.shape[0] == len(d), f"{side} 帧{fi}: 检测{len(d)}点 vs meta{mm.shape[0]}行不对齐"
        # 与 03b --no-foam 同判据：meta 行不足 4 列视为非水沫
        foam = np.array([len(m) >= 4 and m[3] >= 1 for m in mm], dtype=bool)
        keep = foam | (rng.random(len(d)) < p)
        dets_out.append([pt for pt, k in zip(d, keep) if k])
        metas_out.append(mm[keep])
        n_tot += len(d)
        n_foam += int(foam.sum())
        n_keep += int(keep.sum())
        n_keep_nonfoam += int((keep & ~foam).sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_det, "wb") as f:
        pickle.dump(dets_out, f)
    with open(out_meta, "wb") as f:
        pickle.dump(metas_out, f)
    print(f"[thin] {side}: p={p} seed={seed} | 总检测 {n_tot} → 保留 {n_keep}"
          f"（非水沫 {n_keep_nonfoam}/{n_tot - n_foam} ≈ "
          f"{n_keep_nonfoam / max(n_tot - n_foam, 1):.3f}，水沫 {n_foam} 全保留）"
          f" → {out_det}")
    return n_keep


def main():
    p = float(sys.argv[1])
    seed = int(sys.argv[2])
    sides = sys.argv[3:] if len(sys.argv) > 3 else ["left", "right"]
    if sides == ["both"]:
        sides = ["left", "right"]
    for side in sides:
        thin_side(side, p, seed)


if __name__ == "__main__":
    main()
