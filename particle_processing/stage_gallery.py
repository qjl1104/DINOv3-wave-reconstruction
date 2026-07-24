# particle_processing/stage_gallery.py
"""
全链路各阶段结果可视化画廊。产出（data/visualization/）：
  stage1_raw_vs_rect.png      原始左图 vs 矫正图
  stage2_detection.png        矫正图 + 检测泡沫（红圈）
  stage3_tracks2d.png         2D 轨迹叠加（跳切后，按轨迹随机着色）
  stage4_matches.png          跨相机匹配连线（左矫正图 | 右矫正图）
  stage5_pointcloud.png       3D 点云：俯视 X-Y（Z 上色）/ Z 分布 / 某时刻波剖面
用法：../.venv_fs/Scripts/python.exe stage_gallery.py
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
__main__.RobustKalmanFilter = type("RobustKalmanFilter", (rr.SimpleKalmanFilter,), {})

CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")
OUT = os.path.join(ROOT, "data/visualization")
FRAME = 500


def rectify(img, calib, side):
    return cv2.remap(img, calib[f"map1_{side}"], calib[f"map2_{side}"], cv2.INTER_LINEAR)


def stage1(calib):
    raw = cv2.imread(os.path.join(ROOT, "data/left_images/left0001.bmp"), 0)
    rect = rectify(raw, calib, "left")
    both = np.hstack([cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)])
    for txt, x in [("raw left0001", 20), ("rectified", 2580)]:
        cv2.putText(both, txt, (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imwrite(os.path.join(OUT, "stage1_raw_vs_rect.png"), both)


def stage2(calib):
    rect = rectify(cv2.imread(os.path.join(ROOT, "data/left_images/left0001.bmp"), 0), calib, "left")
    _, bw = cv2.threshold(cv2.GaussianBlur(rect, (3, 3), 0), 100, 255, cv2.THRESH_BINARY)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
    vis = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
    k = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 30 <= a <= 5000:
            cv2.circle(vis, tuple(np.int32(cent[i])), 8, (0, 0, 255), 1, cv2.LINE_AA)
            k += 1
    cv2.putText(vis, f"detected blobs: {k}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imwrite(os.path.join(OUT, "stage2_detection.png"), vis)
    print(f"  stage2: {k} blobs")


def stage3(calib):
    with open(os.path.join(ROOT, "data/trajectories/trajectories_2d_left_jumpcut.pkl"), "rb") as f:
        tracks = pickle.load(f)
    rect = cv2.cvtColor(
        rectify(cv2.imread(os.path.join(ROOT, "data/left_images/left0001.bmp"), 0), calib, "left"),
        cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(0)
    for t in tracks:
        fr = sorted(t.points)
        pts = np.array([t.points[k] for k in fr], np.int32)
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        cv2.polylines(rect, [pts], False, col, 1, cv2.LINE_AA)
    cv2.putText(rect, f"2D tracks (jumpcut): {len(tracks)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imwrite(os.path.join(OUT, "stage3_tracks2d.png"), rect)
    print(f"  stage3: {len(tracks)} tracks")


def stage4(calib):
    """取覆盖点最多的一帧，把该帧匹配点投回左右矫正图并连线。"""
    with open(os.path.join(ROOT, "data/trajectories/trajectories_3d_v2_dino.pkl"), "rb") as f:
        trajs = pickle.load(f)
    from collections import Counter
    cnt = Counter()
    for t in trajs:
        for fr in set(t[:, 0].astype(int)):
            cnt[fr] += 1
    frame = cnt.most_common(1)[0][0]
    K_rect = calib["P1"][:, :3]
    t_rect = np.linalg.inv(calib["P2"][:, :3]) @ calib["P2"][:, 3]
    R1 = calib["R1"]
    pts = np.vstack([t[t[:, 0] == frame, 1:4] for t in trajs
                     if np.any(t[:, 0] == frame)])
    xr = pts @ R1.T
    hl = xr @ K_rect.T
    pl = hl[:, :2] / hl[:, 2:3]
    hr = (xr + t_rect) @ K_rect.T
    pr = hr[:, :2] / hr[:, 2:3]
    L = cv2.cvtColor(rectify(cv2.imread(os.path.join(ROOT, f"data/left_images/left{frame + 1:04d}.bmp"), 0), calib, "left"), cv2.COLOR_GRAY2BGR)
    R = cv2.cvtColor(rectify(cv2.imread(os.path.join(ROOT, f"data/right_images/right{frame + 1:04d}.bmp"), 0), calib, "right"), cv2.COLOR_GRAY2BGR)
    both = np.hstack([L, R])
    W = L.shape[1]
    rng = np.random.default_rng(2)
    sel = rng.choice(len(pl), min(40, len(pl)), replace=False)
    for k in sel:
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        p_l = tuple(np.int32(pl[k]))
        p_r = tuple(np.int32(pr[k] + [W, 0]))
        cv2.line(both, p_l, p_r, col, 1, cv2.LINE_AA)
        cv2.circle(both, p_l, 4, col, -1, cv2.LINE_AA)
        cv2.circle(both, p_r, 4, col, -1, cv2.LINE_AA)
    cv2.putText(both, f"matches @frame{frame}: {len(pl)} pts (40 shown)", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imwrite(os.path.join(OUT, "stage4_matches.png"), both)
    print(f"  stage4: {len(pl)} matched pts @f{frame}")


def stage5(calib):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    with open(os.path.join(ROOT, "data/trajectories/trajectories_3d_v2_dino.pkl"), "rb") as f:
        trajs = pickle.load(f)
    pts = np.vstack([t[:, 1:4] for t in trajs if len(t) > 0])
    fr_all = np.concatenate([t[:, 0] for t in trajs if len(t) > 0])
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    sc = ax[0].scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=1, cmap="viridis")
    ax[0].set_title("3D 点云俯视 (X-Y, 颜色=Z mm)")
    ax[0].set_xlabel("X mm"); ax[0].set_ylabel("Y mm"); ax[0].set_aspect("equal")
    fig.colorbar(sc, ax=ax[0])
    ax[1].hist(pts[:, 2], bins=50)
    ax[1].set_title("Z 分布 mm")
    # 某时刻波剖面：取 frame 500±2 的点，沿 PCA 面内传播向 ξ 画 η
    sel = np.abs(fr_all - FRAME) <= 2
    p = pts[sel]
    if len(p) > 10:
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        n = vt[2] * np.sign(vt[2][2])
        u1, u2 = vt[0], vt[1]
        xi = (p - c) @ u2 if abs(u2[1]) > abs(u1[1]) else (p - c) @ u1
        eta = (p - c) @ n
        ax[2].scatter(xi, eta, s=6)
        ax[2].set_title(f"波剖面 frame{FRAME}±2（沿传播向 ξ vs η）")
        ax[2].set_xlabel("ξ mm"); ax[2].set_ylabel("η mm")
    out = os.path.join(OUT, "stage5_pointcloud.png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  stage5: {len(pts)} pts, snapshot {sel.sum()}")


def main():
    os.makedirs(OUT, exist_ok=True)
    calib = np.load(CALIB)
    stage1(calib); stage2(calib); stage3(calib); stage4(calib); stage5(calib)
    print("done")


if __name__ == "__main__":
    main()
