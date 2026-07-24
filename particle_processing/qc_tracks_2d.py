# particle_processing/qc_tracks_2d.py
"""
2D 轨迹质检（QC）：三层判据——统计数、全图叠加、patch 抽查。

产出（data/visualization/）：
  qc_tracks_overlay_{left,right}.png  全部轨迹叠加在矫正图首帧上（平滑点团=好，游走长线=坏）
  qc_tracks_stats_{left,right}.png    长度分布 + 空洞分布 + 最长 6 条的 y(t) 与频谱
  qc_tracks_patches_{left,right}.png  随机抽 6 条轨迹 × 8 帧的 patch 条带（肉眼确认同一泡沫）

用法：../.venv_fs/Scripts/python.exe qc_tracks_2d.py [left|right]（默认 left）
"""

import os
import pickle
import sys

import cv2
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

CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")
OUT_DIR = os.path.join(ROOT, "data/visualization")
FPS = 50.0
F_WAVE = 0.79          # 理论波频 Hz
F_TOL = 0.1            # 主峰容差 ±Hz
MIN_LEN_SHOW = 30


def rectify(img, calib, side):
    return cv2.remap(img, calib[f"map1_{side}"], calib[f"map2_{side}"],
                     cv2.INTER_LINEAR)


def load(side):
    pkl = os.path.join(ROOT, f"data/trajectories/trajectories_2d_{side}_optimized.pkl")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def stats_text(tracks):
    lens = np.array([len(t.points) for t in tracks])
    gaps = []
    for t in tracks:
        fr = np.array(sorted(t.points.keys()))
        g = np.diff(fr) - 1
        gaps += [x for x in g if x > 0]
    lines = [
        f"轨迹条数: {len(tracks)}（≥{MIN_LEN_SHOW} 帧）",
        f"长度: 中位 {np.median(lens):.0f} | p90 {np.percentile(lens, 90):.0f} | 最长 {lens.max()}",
        f"帧空洞: {len(gaps)} 个 | 最大 {max(gaps) if gaps else 0} 帧",
    ]
    return lines, lens, gaps


def fig_overlay(tracks, calib, side):
    img0 = cv2.imread(os.path.join(ROOT, f"data/{side}_images/{side}0001.bmp"), 0)
    rect = cv2.cvtColor(rectify(img0, calib, side), cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(0)
    for t in tracks:
        fr = sorted(t.points.keys())
        pts = np.array([t.points[f] for f in fr], np.int32)
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        cv2.polylines(rect, [pts], False, color, 1, cv2.LINE_AA)
        cv2.circle(rect, tuple(pts[len(pts) // 2]), 3, color, -1, cv2.LINE_AA)
    out = os.path.join(OUT_DIR, f"qc_tracks_overlay_{side}.png")
    cv2.imwrite(out, rect)
    return out


def fig_stats(tracks, side, lens, gaps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    ax[0, 0].hist(lens, bins=30)
    ax[0, 0].set_title("轨迹长度分布（帧）")
    ax[0, 1].hist(gaps if gaps else [0], bins=range(0, max(gaps) + 2 if gaps else 2))
    ax[0, 1].set_title("帧空洞大小分布")

    longest = sorted(tracks, key=lambda t: -len(t.points))[:6]
    # 上排第 3 列叠画 6 条 y(t)；下排 3 列各叠 2 条的频谱
    for t in longest:
        fr = np.array(sorted(t.points.keys()))
        ys = np.array([t.points[f][1] for f in fr])
        ax[0, 2].plot(fr / FPS, ys - ys.mean(), lw=0.8,
                      label=f"id{t.id} ({len(fr)}f)")
    ax[0, 2].set_title("最长 6 条的 y(t)（去均值）")
    ax[0, 2].legend(fontsize=7)
    for k, t in enumerate(longest):
        fr = np.array(sorted(t.points.keys()))
        ys = np.array([t.points[f][1] for f in fr])
        n = len(ys)
        sp = np.abs(np.fft.rfft(ys - ys.mean()))
        fq = np.fft.rfftfreq(n, 1 / FPS)
        pk = np.argmax(sp[1:]) + 1
        hit = "OK" if abs(fq[pk] - F_WAVE) <= F_TOL else "NG"
        axx = ax[1, k % 3]
        axx.plot(fq, sp, lw=0.9, label=f"id{t.id} {hit} {fq[pk]:.2f}Hz")
        axx.axvline(F_WAVE, color="r", ls="--", lw=0.7)
        axx.set_xlim(0, 3)
        axx.legend(fontsize=8)
    for k in range(3):
        ax[1, k].set_xlabel("Hz")
    out = os.path.join(OUT_DIR, f"qc_tracks_stats_{side}.png")
    fig.suptitle(f"2D 轨迹统计（{side}）")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_patches(tracks, calib, side, n_tracks=6, n_frames=8, half=24):
    rng = np.random.default_rng(1)
    pool = [t for t in tracks if len(t.points) >= 100]
    sel = rng.choice(pool, min(n_tracks, len(pool)), replace=False)
    tiles = []
    for t in sel:
        frs = np.linspace(min(t.points), max(t.points), n_frames).astype(int)
        row = []
        for f in frs:
            img = cv2.imread(os.path.join(
                ROOT, f"data/{side}_images/{side}{f + 1:04d}.bmp"), 0)
            if img is None:
                row.append(np.zeros((2 * half, 2 * half), np.uint8))
                continue
            rect = rectify(img, calib, side)
            if f not in t.points:  # 该帧轨迹丢失 → 黑块示意
                row.append(np.zeros((2 * half, 2 * half), np.uint8))
                continue
            x, y = t.points[f]
            x0, y0 = int(round(x)) - half, int(round(y)) - half
            patch = np.zeros((2 * half, 2 * half), np.uint8)
            sy0, sy1 = max(0, y0), min(rect.shape[0], y0 + 2 * half)
            sx0, sx1 = max(0, x0), min(rect.shape[1], x0 + 2 * half)
            patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rect[sy0:sy1, sx0:sx1]
            row.append(patch)
        tiles.append(np.hstack(row))
    sheet = np.vstack(tiles)
    sheet_c = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for i in range(1, n_frames):
        cv2.line(sheet_c, (i * 2 * half, 0), (i * 2 * half, sheet_c.shape[0]),
                 (0, 255, 0), 1)
    out = os.path.join(OUT_DIR, f"qc_tracks_patches_{side}.png")
    cv2.imwrite(out, sheet_c)
    return out, [int(t.id) for t in sel]


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "left"
    os.makedirs(OUT_DIR, exist_ok=True)
    calib = np.load(CALIB)
    tracks = [t for t in load(side) if len(t.points) >= MIN_LEN_SHOW]
    lines, lens, gaps = stats_text(tracks)
    for s in lines:
        print(s)
    o1 = fig_overlay(tracks, calib, side)
    o2 = fig_stats(tracks, side, lens, gaps)
    o3, ids = fig_patches(tracks, calib, side)
    print(f"[输出] {o1}\n[输出] {o2}\n[输出] {o3}（抽查轨迹 id: {ids}）")


if __name__ == "__main__":
    main()
