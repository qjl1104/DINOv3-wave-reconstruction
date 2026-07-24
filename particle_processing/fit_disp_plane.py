# particle_processing/fit_disp_plane.py
"""
视差平面先验系数拟合（bootstrap 助手）。

输入一个"可信"3D 轨迹 pkl（典型地：紧视差窗 pass-1 的产物，匹配基本正确的少量片段），
逐片段：
  1) 3D 点投影回左/右矫正图（K'、R1、t' 全来自标定 npz）；
  2) 在 2D 轨迹 pkl 中按"逐帧投影距离最小"认领产生该片段的左右轨迹；
  3) 实测视差 d = 共同帧上（左x − 右x）中位数；位置 (x,y) = 左轨迹矫正坐标均值。
全部片段的 (x, y, d) 样本用迭代最小二乘（MAD 截尾）拟合 d = a·x + b·y + c，
打印可直接粘贴进 rematch_rectified.py 的 DISP_PLANE = (a, b, c) 及残差统计。

用法：../.venv_fs/Scripts/python.exe fit_disp_plane.py <pkl3d> [traj_l traj_r]
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
           "SimpleKalmanFilter", "ImprovedKalmanFilter", "ExtendedKalmanFilter",
           "OptimizedExtendedKalmanFilter", "UltraOptimizedKalmanFilter",
           "WaveParticleKalmanFilter", "StrictWaveKalmanFilter"]:
    setattr(__main__, _n, getattr(rr, _n))

CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")
TRAJ_L = os.path.join(ROOT, "data/trajectories/trajectories_2d_left_optimized.pkl")
TRAJ_R = os.path.join(ROOT, "data/trajectories/trajectories_2d_right_optimized.pkl")

MIN_COVER = 0.8     # 认领轨迹的帧覆盖率下限
MAX_PROJ_DIST = 3.0  # 认领轨迹的逐帧投影距离中位数上限 px（重投影噪声 ~1.5px）


def project_rect(xyz, K_rect, R1, t_rect):
    """相机1系 3D 点 → 左/右矫正图像素。"""
    xr = xyz @ R1.T                      # X_rect = R1 @ X_cam1（行向量）
    hl = xr @ K_rect.T
    pl = hl[:, :2] / hl[:, 2:3]
    hr = (xr + t_rect) @ K_rect.T        # 右矫正系：X_rect + t'（t'=(-B,0,0)）
    pr = hr[:, :2] / hr[:, 2:3]
    return pl, pr


def claim_track(proj, frames, tracks):
    """在 tracks 中认领产生投影 proj 的轨迹：帧覆盖率 ≥ MIN_COVER 且
    逐帧距离中位数最小、< MAX_PROJ_DIST。返回 (track, med_dist) 或 None。"""
    best, best_d = None, np.inf
    for t in tracks:
        have = [k for k, f in enumerate(frames) if f in t.points]
        if len(have) < MIN_COVER * len(frames):
            continue
        d = np.median([np.hypot(t.points[frames[k]][0] - proj[k, 0],
                                t.points[frames[k]][1] - proj[k, 1]) for k in have])
        if d < best_d:
            best, best_d = t, d
    if best is not None and best_d < MAX_PROJ_DIST:
        return best, best_d
    return None


def fit_plane(samples):
    """(x, y, d) 样本 → 迭代 LS（MAD 截尾，最多 4 轮）。返回 (coef, res, keep)。"""
    P = np.array([[x, y, 1.0] for x, y, _d in samples])
    d = np.array([s[2] for s in samples])
    keep = np.ones(len(d), dtype=bool)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(P[keep], d[keep], rcond=None)
        res = d - P @ coef
        med = np.median(res[keep])
        mad = 1.4826 * np.median(np.abs(res[keep] - med))
        thr = max(3.0 * mad, 1.0)
        new_keep = np.abs(res - med) <= thr
        if (new_keep == keep).all():
            break
        keep = new_keep
    return coef, res, keep


def main():
    pkl3d = sys.argv[1] if len(sys.argv) > 1 else None
    if not pkl3d:
        sys.exit("用法：fit_disp_plane.py <pkl3d> [traj_l traj_r]")
    traj_l_path = sys.argv[2] if len(sys.argv) > 2 else TRAJ_L
    traj_r_path = sys.argv[3] if len(sys.argv) > 3 else TRAJ_R

    calib = np.load(CALIB)
    P1, P2, R1 = calib["P1"], calib["P2"], calib["R1"]
    K_rect = P1[:, :3]
    t_rect = np.linalg.inv(P2[:, :3]) @ P2[:, 3]
    f_px, B = K_rect[0, 0], abs(t_rect[0])

    frags = pickle.load(open(pkl3d, "rb"))
    tracks_l = pickle.load(open(traj_l_path, "rb"))
    tracks_r = pickle.load(open(traj_r_path, "rb"))
    tracks_l = [t for t in tracks_l if len(t.points) >= rr.MIN_TRAJ_LEN]
    tracks_r = [t for t in tracks_r if len(t.points) >= rr.MIN_TRAJ_LEN]
    print(f"3D 片段 {len(frags)} 条；2D 轨迹 左 {len(tracks_l)} / 右 {len(tracks_r)} 条")

    samples = []
    for k, frag in enumerate(frags):
        frag = np.asarray(frag, dtype=np.float64)
        frames = frag[:, 0].astype(int)
        pl, pr = project_rect(frag[:, 1:4], K_rect, R1, t_rect)
        cl = claim_track(pl, frames, tracks_l)
        cr = claim_track(pr, frames, tracks_r)
        if cl is None or cr is None:
            print(f"  片段 {k}（{len(frames)} 帧）：认领失败 "
              f"（左 {'%.2fpx' % cl[1] if cl else '无'}，右 {'%.2fpx' % cr[1] if cr else '无'}），跳过")
            continue
        tl, tr = cl[0], cr[0]
        common = [f for f in frames if f in tl.points and f in tr.points]
        disp = np.median([tl.points[f][0] - tr.points[f][0] for f in common])
        mx = np.mean([tl.points[f][0] for f in common])
        my = np.mean([tl.points[f][1] for f in common])
        samples.append((mx, my, disp))
        print(f"  片段 {k}：({mx:.0f},{my:.0f}) d={disp:.1f}px "
              f"（投影认领 左 {cl[1]:.2f}px 右 {cr[1]:.2f}px，{len(common)} 帧）")
    if len(samples) < 3:
        sys.exit(f"有效样本 {len(samples)} < 3，无法拟合平面")

    coef, res, keep = fit_plane(samples)
    a, b, c = coef
    print(f"\n拟合 {int(keep.sum())}/{len(samples)} 样本：")
    for s, r, kp in zip(samples, res, keep):
        print(f"  ({s[0]:7.0f},{s[1]:7.0f}) d={s[2]:7.1f} pred={a * s[0] + b * s[1] + c:7.1f} "
              f"res={r:+6.1f}{'  [剔除]' if not kp else ''}")
    rk = res[keep]
    print(f"残差：med|r| {np.median(np.abs(rk)):.1f}px | p90 {np.percentile(np.abs(rk), 90):.1f} | "
          f"max {np.abs(rk).max():.1f}")
    ds = np.array([s[2] for s in samples])
    print(f"样本视差范围 {ds.min():.0f}–{ds.max():.0f}px → 深度 "
          f"{f_px * B / ds.max():.0f}–{f_px * B / ds.min():.0f} mm")
    print(f"\n粘贴到 rematch_rectified.py：")
    print(f"DISP_PLANE = ({a:.6e}, {b:.6e}, {c:.6f})")


if __name__ == "__main__":
    main()
