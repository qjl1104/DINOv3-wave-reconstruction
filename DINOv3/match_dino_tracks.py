# DINOv3/match_dino_tracks.py
"""
【已归档 ARCHIVED】尽管名字带 dino，本脚本的匹配判据不含任何 DINO 相似度
（纯 dy/视差几何 + 匈牙利全局指派）；产出 trajectories_3d_dino.pkl 在生产链中
无下游消费者，仅作实验记录保留。

实验A 第二步：DINOv3 描述子轨迹的跨相机匹配 + 三角化。

与 rematch_rectified.py 同原理（矫正坐标下 dy≈0 + 视差近恒定 + 匈牙利/
片段级接受），差别：
- 输入轨迹已在 paper_params_recalculated.npz 的矫正坐标系（feature_cache
  同源），无需 undistortPoints 步骤；
- 三角化用该标定的 Q 矩阵反投影（与 DINOv3 线一致）。

输出：trajectories_3d_dino.pkl —— list[(N,4)] = [frame, X, Y, Z] mm
用法：../.venv_fs/Scripts/python.exe match_dino_tracks.py
"""

import os
import pickle
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config  # noqa: E402
from track_descriptor import Track  # noqa: E402,F401  # unpickle 需要类定义
from utils import reproject_to_3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_OVERLAP = 20
MAX_MED_DY = 3.0
DISP_RANGE = (150, 1200)   # paper_params 坐标系：bf = 3.7187e6，disp 190–1180 ↔ Z ≈ 3.15–19.6 m
MAX_DISP_STD = 40.0
MIN_TRAJ_LEN = 20
DEPTH_RANGE = (2500, 12000)
MAX_EXTENT_UV = 1500.0
ACCEPT_MED_DY = 3.0


def match_pairs(tracks_l, tracks_r):
    fl_all = [sorted(t.points.keys()) for t in tracks_l]
    fr_all = [sorted(t.points.keys()) for t in tracks_r]
    nL, nR = len(tracks_l), len(tracks_r)
    cost = np.full((nL, nR), np.inf)
    stats_grid = {}
    for i, t_l in enumerate(tracks_l):
        fset_l = set(fl_all[i])
        for j, t_r in enumerate(tracks_r):
            common = sorted(fset_l & set(fr_all[j]))
            if len(common) < MIN_OVERLAP:
                continue
            dy = np.array([t_l.points[f][1] - t_r.points[f][1] for f in common])
            disp = np.array([t_l.points[f][0] - t_r.points[f][0] for f in common])
            med_dy, med_disp, std_disp = np.median(np.abs(dy)), np.median(disp), disp.std()
            if med_dy > MAX_MED_DY:
                continue
            if not (DISP_RANGE[0] <= med_disp <= DISP_RANGE[1]):
                continue
            if std_disp > MAX_DISP_STD:
                continue
            cost[i, j] = med_dy + 0.05 * std_disp
            stats_grid[(i, j)] = (len(common), med_dy, med_disp, std_disp)
    # 稠密轨迹下"dy≈0 + 视差近恒定"失去选择性（每个左轨迹都有许多右轨迹
    # 满足条件），片段级接受会把粒子 A 的左轨迹错配给粒子 B 的右轨迹——
    # 错误的三角化仍产生 0.79Hz 振荡（近平面掩蔽），但相位关系全错。
    # 必须用匈牙利 1-to-1 全局指派消灭交叉配对。
    BIG = 1e4
    ok_r = np.isfinite(cost).any(axis=1)
    ok_c = np.isfinite(cost).any(axis=0)
    hungarian = []
    if ok_r.any() and ok_c.any():
        sub = np.where(np.isfinite(cost), cost, BIG)[np.ix_(ok_r, ok_c)]
        rr, cc = linear_sum_assignment(sub)
        ridx, cidx = np.where(ok_r)[0], np.where(ok_c)[0]
        for r, c in zip(rr, cc):
            if sub[r, c] < BIG:
                i, j = ridx[r], cidx[c]
                hungarian.append((i, j, stats_grid[(i, j)]))
    candidates = [(i, j, st) for (i, j), st in stats_grid.items()]
    return hungarian, candidates


def main():
    tracks_l = pickle.load(open(os.path.join(HERE, "data_dino_tracks_left.pkl"), "rb"))
    tracks_r = pickle.load(open(os.path.join(HERE, "data_dino_tracks_right.pkl"), "rb"))
    tracks_l = [t for t in tracks_l if len(t.points) >= MIN_TRAJ_LEN]
    tracks_r = [t for t in tracks_r if len(t.points) >= MIN_TRAJ_LEN]
    print(f"参与匹配：左 {len(tracks_l)} 条，右 {len(tracks_r)} 条")

    Q = np.load(Config().CALIBRATION_FILE)["Q"]
    hungarian, candidates = match_pairs(tracks_l, tracks_r)
    pairs = [p for p in sorted(hungarian, key=lambda p: p[2][1])
             if p[2][1] <= ACCEPT_MED_DY]
    print(f"硬过滤候选 {len(candidates)} 对 | 匈牙利 1-to-1 指派 {len(hungarian)} 对 | "
          f"|dy|≤{ACCEPT_MED_DY}px 接受 {len(pairs)} 对")
    if pairs:
        st = np.array([p[2] for p in pairs])
        print(f"  共同帧 med={int(np.median(st[:, 0]))} | |dy|中位 med="
              f"{np.median(st[:, 1]):.2f}px | 视差 med={np.median(st[:, 2]):.0f}px "
              f"| 视差std med={np.median(st[:, 3]):.1f}px")

    trajs_3d = []
    for iL, jR, _st in pairs:
        t_l, t_r = tracks_l[iL], tracks_r[jR]
        common = sorted(set(t_l.points.keys()) & set(t_r.points.keys()))
        pl = np.array([t_l.points[f] for f in common])
        disp = np.array([t_l.points[f][0] - t_r.points[f][0] for f in common])
        xyz = reproject_to_3d(pl, disp, Q).astype(np.float64)
        trajs_3d.append(np.c_[common, xyz])

    kept = []
    for tr in trajs_3d:
        z_med = np.median(tr[:, 3])
        extent = max(np.ptp(tr[:, 1]), np.ptp(tr[:, 2]))
        if DEPTH_RANGE[0] <= z_med <= DEPTH_RANGE[1] and extent <= MAX_EXTENT_UV:
            kept.append(tr)
    print(f"三角化后质量过滤：{len(kept)}/{len(trajs_3d)} 条保留")
    if kept:
        allpts = np.vstack(kept)
        lens = np.array([len(tr) for tr in kept])
        print(f"  总点数 {len(allpts)} | 片段长度 中位 {np.median(lens):.0f} "
              f"p90 {np.percentile(lens, 90):.0f} max {lens.max()} | "
              f"Z med {np.median(allpts[:, 3]):.0f} mm")

    out = os.path.join(HERE, "trajectories_3d_dino.pkl")
    with open(out, "wb") as f:
        pickle.dump(kept, f)
    print(f"[输出] {out}")


if __name__ == "__main__":
    main()
