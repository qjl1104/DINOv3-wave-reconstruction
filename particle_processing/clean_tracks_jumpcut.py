# particle_processing/clean_tracks_jumpcut.py
"""
2D 轨迹跳切清洗：在物理不可能的跳变处把轨迹切成身份干净的段。

背景：跟踪器（03）在断档滑行后重捕时会抓到相邻泡沫（dist_thresh=80px >
泡沫间距 ~50px），导致单条轨迹混入多个身份——相邻帧跳变 69–304px
（泡沫轨道速度物理上限 ~2px/帧）。下游 3D 链的视差波动门只能拒掉
串号最狠的，≤30px 的小跳号仍漏在轨迹里。

切分判据：相邻观测点的步长 > max(STEP_ABS, STEP_PER_FRAME * 间隔帧数)
即在该处切开；切后长度 < MIN_SEG 的段丢弃（太短对匹配无用且噪声大）。

输入：data/trajectories/trajectories_2d_{side}_optimized.pkl（canonical，只读）
输出：data/trajectories/trajectories_2d_{side}_jumpcut.pkl（新文件）

用法：../.venv_fs/Scripts/python.exe clean_tracks_jumpcut.py [left|right|both]
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rematch_rectified as rr  # noqa: E402
import __main__  # noqa: E402

for _n in ["Track", "UltraTrack", "WaveParticleTrack", "StrictTrack",
           "SimpleKalmanFilter", "ImprovedKalmanFilter",
           "ExtendedKalmanFilter", "OptimizedExtendedKalmanFilter",
           "UltraOptimizedKalmanFilter", "WaveParticleKalmanFilter",
           "StrictWaveKalmanFilter"]:
    setattr(__main__, _n, getattr(rr, _n))
__main__.RobustKalmanFilter = type(
    "RobustKalmanFilter", (rr.SimpleKalmanFilter,), {})

STEP_ABS = 12.0        # 单帧步长硬上限 px（波轨道 ~2px/帧，留足余量）
STEP_PER_FRAME = 4.0   # 断档跨帧时按间隔放宽的每帧上限 px/帧
MIN_SEG = 20           # 切后段长下限（与 rematch 的 MIN_OVERLAP 对齐）


def clean_side(side, src=None, dst=None):
    if src is None:
        src = os.path.join(ROOT, f"data/trajectories/trajectories_2d_{side}_optimized.pkl")
    if dst is None:
        dst = os.path.join(ROOT, f"data/trajectories/trajectories_2d_{side}_jumpcut.pkl")
    with open(src, "rb") as f:
        tracks = pickle.load(f)

    out, n_cut, n_drop = [], 0, 0
    for t in tracks:
        fr = sorted(t.points)
        # 找切点：当前段切分成 [seg_start..] 若干段
        segs, start = [], 0
        for k in range(1, len(fr)):
            a, b = fr[k - 1], fr[k]
            d = np.hypot(t.points[b][0] - t.points[a][0],
                         t.points[b][1] - t.points[a][1])
            if d > max(STEP_ABS, STEP_PER_FRAME * (b - a)):
                segs.append(fr[start:k])
                start = k
                n_cut += 1
        segs.append(fr[start:])
        for seg in segs:
            if len(seg) < MIN_SEG:
                n_drop += 1
                continue
            nt = rr.Track(track_id=t.id)
            nt.points = {f: t.points[f] for f in seg}
            out.append(nt)

    with open(dst, "wb") as f:
        pickle.dump(out, f)
    lens = np.array([len(t.points) for t in out])
    print(f"[{side}] {len(tracks)} 条 → 跳切 {n_cut} 刀 → {len(out)} 段"
          f"（丢弃 <{MIN_SEG} 帧碎段 {n_drop} 个）")
    print(f"      段长：中位 {np.median(lens):.0f} | p90 "
          f"{np.percentile(lens, 90):.0f} | 最长 {lens.max()} → {dst}")
    return len(tracks), len(out)


def main():
    # 用法 A（原样）: clean_tracks_jumpcut.py [left|right|both]
    # 用法 B（自定义路径）: clean_tracks_jumpcut.py --src SRC_L SRC_R --dst DST_L DST_R
    if "--src" in sys.argv:
        i = sys.argv.index("--src")
        srcs = sys.argv[i + 1:i + 3]
        j = sys.argv.index("--dst")
        dsts = sys.argv[j + 1:j + 3]
        for side, src, dst in zip(["left", "right"], srcs, dsts):
            clean_side(side, src, dst)
        return
    sides = sys.argv[1:] if len(sys.argv) > 1 else ["left", "right"]
    if sides == ["both"]:
        sides = ["left", "right"]
    for side in sides:
        clean_side(side)


if __name__ == "__main__":
    main()
