# compare_chain_v2det.py
# -*- coding: utf-8 -*-
"""
v2 检测链 vs 生产链的对比裁判汇总。

对两组 3D 轨迹 pkl 各跑一遍 eval_tracks 指标（子进程），并并排打印：
  生产: data/trajectories/trajectories_3d_v2_dino.pkl
  v2:   data/trajectories/trajectories_3d_v2det_dino.pkl
另外对比 2D 轨迹阶段的数量统计（jumpcut 后段数/长度）。

用法: .venv_fs/Scripts/python.exe compare_chain_v2det.py
"""
import os
import pickle
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

PAIRS = [
    ("生产 (严格检测)", os.path.join(ROOT, "data/trajectories/trajectories_3d_v2_dino.pkl")),
    ("v2 (宽松检测)", os.path.join(ROOT, "data/trajectories/trajectories_3d_v2det_dino.pkl")),
    ("v2+dt40 (宽松检测+收紧关联门)", os.path.join(ROOT, "data/trajectories/trajectories_3d_v2det40_dino.pkl")),
]


def run_eval(pkl, label):
    r = subprocess.run([PY, os.path.join(ROOT, "wave_modeling/eval_tracks.py"), pkl, label],
                       capture_output=True, text=True, cwd=ROOT)
    return r.stdout + (r.stderr if r.returncode else "")


def tracks2d_stats(pkl):
    sys.path.insert(0, os.path.join(ROOT, "particle_processing"))
    import rematch_rectified as rr
    import __main__
    for _n in ["Track", "UltraTrack", "WaveParticleTrack", "StrictTrack",
               "SimpleKalmanFilter", "ImprovedKalmanFilter",
               "ExtendedKalmanFilter", "OptimizedExtendedKalmanFilter",
               "UltraOptimizedKalmanFilter", "WaveParticleKalmanFilter",
               "StrictWaveKalmanFilter"]:
        setattr(__main__, _n, getattr(rr, _n))
    __main__.RobustKalmanFilter = type("RobustKalmanFilter", (rr.SimpleKalmanFilter,), {})
    with open(pkl, "rb") as f:
        tracks = pickle.load(f)
    lens = np.array([len(t.points) for t in tracks])
    return len(tracks), np.median(lens), np.percentile(lens, 90), lens.max()


def main():
    print("=" * 66)
    print("2D 轨迹阶段（jumpcut 后）")
    print("=" * 66)
    for label, side in [("生产", "jumpcut"), ("v2", "v2det_jumpcut"), ("v2+dt40", "v2det40_jumpcut")]:
        for s in ("left", "right"):
            p = os.path.join(ROOT, f"data/trajectories/trajectories_2d_{s}_{side}.pkl")
            if not os.path.exists(p):
                print(f"  {label} {s}: 缺失 {p}")
                continue
            n, med, p90, mx = tracks2d_stats(p)
            print(f"  {label} {s:5s}: {n:4d} 段 | 段长 中位 {med:.0f} p90 {p90:.0f} max {mx}")

    print()
    print("=" * 66)
    print("3D 轨迹裁判（eval_tracks 同一杆秤）")
    print("=" * 66)
    for label, pkl in PAIRS:
        if not os.path.exists(pkl):
            print(f"\n### {label}: 缺失 {pkl}")
            continue
        print(f"\n### {label} ###")
        print(run_eval(pkl, label))


if __name__ == "__main__":
    main()
