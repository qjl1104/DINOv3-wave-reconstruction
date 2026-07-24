# particle_processing/qc_tracks_video.py
"""
2D 轨迹动态质检：跟踪视频 + 覆盖栅格图。

产出（data/visualization/）：
  qc_tracks_{side}.mp4          跟踪叠加视频（1280×800@30fps）：
                                彩色点=当前位置，同色尾巴=近 30 帧轨迹，
                                绿色圈=轨迹出生帧，红叉=死亡帧，
                                黄色虚线=断档期两端（丢失期间的"断桥"），
                                长轨迹显示 id，左上角 HUD 帧号/时间/活跃数
  qc_tracks_coverage_{side}.png 覆盖栅格：每行一条轨迹（按出生帧排序），
                                白=在册，红=寿命期内的断档，黑=未出生/已死亡

打印：最长的 20 个断档事件（轨迹 id / 帧区间 / 长度），可在视频中定位。

用法：../.venv_fs/Scripts/python.exe qc_tracks_video.py [left|right] [起始帧 帧数]
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
TAIL = 30               # 尾巴长度（帧）
LABEL_MIN_LEN = 300     # ≥此长度才在视频里显示 id
SCALE = 0.5             # 视频缩放（2560×1600 → 1280×800）


def color_of(i):
    rng = np.random.default_rng(i * 7919)
    return tuple(int(c) for c in rng.integers(70, 255, 3))


def dashed_line(img, p1, p2, color, dash=6):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    n = int(np.linalg.norm(p2 - p1) / dash) + 1
    for k in range(0, n, 2):
        a = p1 + (p2 - p1) * k / n
        b = p1 + (p2 - p1) * min(k + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, 1, cv2.LINE_AA)


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "left"
    f_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    calib = np.load(CALIB)
    with open(os.path.join(ROOT, f"data/trajectories/trajectories_2d_{side}_optimized.pkl"), "rb") as f:
        tracks = pickle.load(f)

    birth = {i: min(t.points) for i, t in enumerate(tracks)}
    death = {i: max(t.points) for i, t in enumerate(tracks)}
    # 每条轨迹的断档区间 [(g0, g1)]：g0 < f < g1 缺失
    gaps_of = {}
    for i, t in enumerate(tracks):
        fr = sorted(t.points)
        g = [(a + 1, b - 1) for a, b in zip(fr[:-1], fr[1:]) if b - a > 1]
        if g:
            gaps_of[i] = g

    # ---- 覆盖栅格 ----
    order = sorted(range(len(tracks)), key=lambda i: (birth[i], death[i]))
    raster = np.zeros((len(order), 1000, 3), np.uint8)
    for r, i in enumerate(order):
        t = tracks[i]
        for f in range(birth[i], death[i] + 1):
            raster[r, f] = (200, 200, 200) if f in t.points else (0, 0, 220)
    raster = cv2.resize(raster, (2000, max(200, len(order) * 2)),
                        interpolation=cv2.INTER_NEAREST)
    cv2.putText(raster, "white=present red=gap (rows=tracks by birth, cols=frame 0-999)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    out_png = os.path.join(OUT_DIR, f"qc_tracks_coverage_{side}.png")
    cv2.imwrite(out_png, raster)

    # ---- 断档事件打印 ----
    events = sorted(((g1 - g0 + 1, i, g0, g1)
                     for i, gs in gaps_of.items() for g0, g1 in gs), reverse=True)
    print("最长 20 个断档（长度, 轨迹id, 帧 g0-g1）:")
    for L, i, g0, g1 in events[:20]:
        print(f"  {L:3d}帧  id={tracks[i].id:<6} 帧 {g0}-{g1}（轨迹总长 {len(tracks[i].points)}）")

    # ---- 视频 ----
    out_mp4 = os.path.join(OUT_DIR, f"qc_tracks_{side}.mp4")
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                         (int(2560 * SCALE), int(1600 * SCALE)))
    for f in range(f_start, min(f_start + n_frames, 1000)):
        img = cv2.imread(os.path.join(ROOT, f"data/{side}_images/{side}{f + 1:04d}.bmp"), 0)
        rect = cv2.remap(img, calib[f"map1_{side}"], calib[f"map2_{side}"], cv2.INTER_LINEAR)
        vis = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
        n_active = 0
        for i, t in enumerate(tracks):
            if not (birth[i] <= f <= death[i]):
                continue
            col = color_of(i)
            # 断档"断桥"：本帧处于断档内 → 画两端点虚线
            if f not in t.points:
                for g0, g1 in gaps_of.get(i, []):
                    if g0 <= f <= g1:
                        dashed_line(vis, t.points[g0 - 1], t.points[g1 + 1], (0, 220, 220))
                continue
            n_active += 1
            p = tuple(int(round(v)) for v in t.points[f])
            tail = [t.points[k] for k in range(max(birth[i], f - TAIL), f + 1) if k in t.points]
            if len(tail) > 1:
                cv2.polylines(vis, [np.array(tail, np.int32)], False, col, 1, cv2.LINE_AA)
            cv2.circle(vis, p, 3, col, -1, cv2.LINE_AA)
            if f == birth[i]:
                cv2.circle(vis, p, 8, (0, 255, 0), 2, cv2.LINE_AA)   # 出生：绿圈
            if f == death[i]:
                cv2.drawMarker(vis, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)  # 死亡：红叉
            if len(t.points) >= LABEL_MIN_LEN:
                cv2.putText(vis, str(t.id), (p[0] + 5, p[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        cv2.putText(vis, f"frame {f}  t={f / FPS:.2f}s  active={n_active}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        vis = cv2.resize(vis, None, fx=SCALE, fy=SCALE)
        vw.write(vis)
        if (f - f_start + 1) % 200 == 0:
            print(f"  视频 {f - f_start + 1}/{n_frames}")
    vw.release()
    print(f"[输出] {out_mp4}\n[输出] {out_png}")


if __name__ == "__main__":
    main()
