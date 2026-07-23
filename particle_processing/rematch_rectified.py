# particle_processing/rematch_rectified.py
"""
替代 04_trajectory_matching.py + 05_reconstruction_3d.py 的修正版立体匹配与三角测量。

修正的三个问题：
1. 旧 04 的"左右轨迹起止点像素距离 < 200px"预过滤在物理上错误——
   该双目系统基线 1413mm，矫正后视差本身就有 400–1000px，
   正确匹配全被该过滤误杀，只剩 19 对且多对错配（3D 深度 3–29m 发散）。
   本脚本改为：先把 2D 点矫正到 rectified 坐标系（undistortPoints + R1/R2/P1/P2），
   此时正确匹配必然满足 同行（dy≈0）、视差为正且大致恒定——用这两条做匹配。
2. 2D 轨迹本来就带绝对帧号（points dict 的 key），无需 DTW，
   直接按帧对齐逐点比较。
3. 保留每个 3D 点的绝对帧号（旧 05 丢弃了），输出 [frame, x, y, z]。

输出：data/trajectories/trajectories_3d_v2.pkl
    list[np.ndarray]，每条轨迹 shape (N, 4)：frame, X, Y, Z（单位随标定，mm）
诊断图：data/trajectories/rematch_diagnostic.png
"""

import os
import pickle

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ_L = os.path.join(ROOT, "data/trajectories/trajectories_2d_left_optimized.pkl")
TRAJ_R = os.path.join(ROOT, "data/trajectories/trajectories_2d_right_optimized.pkl")
CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")
OUT_PKL = os.path.join(ROOT, "data/trajectories/trajectories_3d_v2.pkl")
OUT_PNG = os.path.join(ROOT, "data/trajectories/rematch_diagnostic.png")

# 匹配阈值（矫正坐标系，像素）
MIN_OVERLAP = 20        # 最少共同帧数
MAX_MED_DY = 3.0        # 矫正后纵向偏差中位数上限
DISP_RANGE = (300, 1100)  # 视差合理范围（FoundationStereo 实测 413–1001 余量放宽）
MAX_DISP_STD = 30.0     # 视差波动上限（波幅小 → 深度近似不变 → 视差近恒定）
MIN_TRAJ_LEN = 30       # 参与匹配的最短轨迹
DEPTH_RANGE = (2500, 10000)  # 三角化后中位深度合理范围 mm（FS 实测 3.7–9m 加余量）
MAX_EXTENT_UV = 1500.0  # 单条轨迹水平跨度上限 mm（防 ID 串接）
ACCEPT_MED_DY = 3.0     # 片段级接受的矫正 |dy| 中位数上限 px（矫正噪声 ~1-2px，仍很严格）


# ---------------- pickle 兼容桩类（轨迹对象的类定义在各 03 脚本里） ----------------
class Track:
    def __init__(self, track_id=None, initial_detection=None, frame_idx=None):
        self.id = track_id
        self.points = {}
        if initial_detection is not None:
            self.points = {frame_idx: initial_detection}

    def get_ordered_points(self):
        return [self.points[fi] for fi in sorted(self.points.keys())]


class UltraTrack(Track):
    pass


class WaveParticleTrack(Track):
    pass


class StrictTrack(Track):
    pass


class SimpleKalmanFilter:
    def __init__(self, initial_pos=None):
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.Q = np.eye(4)
        self.R = np.eye(2)
        self.dt = 1.0


class ImprovedKalmanFilter(SimpleKalmanFilter):
    pass


class ExtendedKalmanFilter(SimpleKalmanFilter):
    pass


class OptimizedExtendedKalmanFilter(SimpleKalmanFilter):
    pass


class UltraOptimizedKalmanFilter(SimpleKalmanFilter):
    pass


class WaveParticleKalmanFilter(SimpleKalmanFilter):
    pass


class StrictWaveKalmanFilter(SimpleKalmanFilter):
    pass


# ----------------------------------------------------------------------

def rectify_traj(traj, K, D, R_rect, P_rect):
    """轨迹 {frame: (x,y)} → 矫正坐标系下的 {frame: (x',y')}"""
    frames = sorted(traj.points.keys())
    pts = np.array([traj.points[f] for f in frames], dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, D, R=R_rect, P=P_rect).reshape(-1, 2)
    return frames, out


def match_pairs(rect_left, rect_right):
    """rect_*: list of (frames, pts[N,2])。返回 [(iL, jR, stats)]。
    成本 = 共同帧上 |dy| 的中位数；硬过滤：视差范围与波动。"""
    nL, nR = len(rect_left), len(rect_right)
    pos_l = [{f: k for k, f in enumerate(fl)} for fl, _ in rect_left]
    pos_r = [{f: k for k, f in enumerate(fr)} for fr, _ in rect_right]
    cost = np.full((nL, nR), np.inf)
    stats_grid = {}
    for i, (fl, pl) in enumerate(rect_left):
        fset_l = set(fl)
        for j, (fr, pr) in enumerate(rect_right):
            common = fset_l & set(fr)
            if len(common) < MIN_OVERLAP:
                continue
            il = [pos_l[i][f] for f in common]
            ir = [pos_r[j][f] for f in common]
            dy = pl[il, 1] - pr[ir, 1]
            disp = pl[il, 0] - pr[ir, 0]
            med_dy, med_disp, std_disp = np.median(np.abs(dy)), np.median(disp), disp.std()
            if med_dy > MAX_MED_DY:
                continue
            if not (DISP_RANGE[0] <= med_disp <= DISP_RANGE[1]):
                continue
            if std_disp > MAX_DISP_STD:
                continue
            cost[i, j] = med_dy + 0.05 * std_disp
            stats_grid[(i, j)] = (len(common), med_dy, med_disp, std_disp)
    # 用大数代替 inf 再做指派（inf 在"两行只有一个共同可行列"时会判 infeasible），
    # 指派后按真实成本阈值过滤。
    # 注意：匈牙利 1-to-1 结果仅用于日志对照，不参与接受决策——
    # 实际接受走下方 candidates（片段级，故意允许多对多：
    # 同一粒子断成的多个片段对各自给出独立 3D 采样，重复点由下游按帧去重）。
    BIG = 1e4
    ok_r = np.isfinite(cost).any(axis=1)
    ok_c = np.isfinite(cost).any(axis=0)
    hungarian = []
    if ok_r.any() and ok_c.any():
        sub = cost[np.ix_(ok_r, ok_c)]
        sub = np.where(np.isfinite(sub), sub, BIG)
        rr, cc = linear_sum_assignment(sub)
        ridx, cidx = np.where(ok_r)[0], np.where(ok_c)[0]
        for r, c in zip(rr, cc):
            if sub[r, c] < BIG:
                i, j = ridx[r], cidx[c]
                hungarian.append((i, j, stats_grid[(i, j)]))
    # 全部通过硬过滤的候选对（片段级，未经 1-to-1 约束）
    candidates = [(i, j, st) for (i, j), st in stats_grid.items()]
    return hungarian, candidates


def triangulate_pairs(pairs, rect_left, rect_right, raw_left, raw_right, P1, P2):
    """对匹配对按共同帧三角化（标准 OpenCV 矫正双目三角化）：
    1) undistortPoints 带 R1/R2 → 左右点成为"矫正坐标系"下的归一化光线；
    2) 用矫正投影的归一化形式 P1r=[I|0]、P2r=[I|t'] 三角化
       （t' 由矫正 P2 提取：P2=K'[I|t']，t'=inv(K')@P2[:,3]，≈(-B,0,0)）；
    3) 三角化结果在矫正左目坐标系，左乘 R1.T 转回原左目坐标系输出。
    （旧实现去畸变不带 R1/R2——光线留在原始相机系——却配矫正系的 P 阵，
    两个坐标系差 ~8.5°，合成数据往返误差中位 ~26m，属坐标系混用。）
    R1/R2 与 KL/DL/KR/DR 同为模块级全局（__main__ 块或调用方赋值）。"""
    t_rect = np.linalg.inv(P2[:, :3]) @ P2[:, 3]
    P1r = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2r = np.hstack([np.eye(3), t_rect.reshape(3, 1)])
    trajs_3d = []
    for iL, jR, _st in pairs:
        fl, _ = rect_left[iL]
        fr, _ = rect_right[jR]
        common = sorted(set(fl) & set(fr))
        pl = np.array([raw_left[iL].points[f] for f in common], dtype=np.float64)
        pr = np.array([raw_right[jR].points[f] for f in common], dtype=np.float64)
        pl_n = cv2.undistortPoints(pl.reshape(-1, 1, 2), KL, DL, R=R1).reshape(-1, 2)
        pr_n = cv2.undistortPoints(pr.reshape(-1, 1, 2), KR, DR, R=R2).reshape(-1, 2)
        pts4 = cv2.triangulatePoints(P1r, P2r, pl_n.T, pr_n.T)
        xyz = (pts4[:3] / pts4[3]).T
        xyz = xyz @ R1  # 行向量写法：X_cam1 = R1.T @ X_rect
        trajs_3d.append(np.c_[common, xyz])
    return trajs_3d


if __name__ == "__main__":
    import sys
    # 可选命令行覆盖输入/输出（默认验证过的 *_optimized.pkl → trajectories_3d_v2.pkl）
    traj_l_path = sys.argv[1] if len(sys.argv) > 1 else TRAJ_L
    traj_r_path = sys.argv[2] if len(sys.argv) > 2 else TRAJ_R
    out_pkl = sys.argv[3] if len(sys.argv) > 3 else OUT_PKL
    out_png = sys.argv[4] if len(sys.argv) > 4 else OUT_PNG
    with open(traj_l_path, "rb") as f:
        raw_left = pickle.load(f)
    with open(traj_r_path, "rb") as f:
        raw_right = pickle.load(f)
    calib = np.load(CALIB)
    KL, DL = calib["K_left"], calib["D_left"].ravel()
    KR, DR = calib["K_right"], calib["D_right"].ravel()
    R1, R2, P1, P2 = calib["R1"], calib["R2"], calib["P1"], calib["P2"]

    raw_left = [t for t in raw_left if len(t.points) >= MIN_TRAJ_LEN]
    raw_right = [t for t in raw_right if len(t.points) >= MIN_TRAJ_LEN]
    print(f"参与匹配：左 {len(raw_left)} 条，右 {len(raw_right)} 条")

    rect_left = [rectify_traj(t, KL, DL, R1, P1) for t in raw_left]
    rect_right = [rectify_traj(t, KR, DR, R2, P2) for t in raw_right]

    hungarian, candidates = match_pairs(rect_left, rect_right)
    print(f"匈牙利 1-to-1 匹配 {len(hungarian)} 对；硬过滤全部候选 {len(candidates)} 对")
    if hungarian:
        st = np.array([p[2] for p in hungarian])
        print(f"  [匈牙利] 共同帧 med={int(np.median(st[:,0]))} | "
              f"|dy|中位 med={np.median(st[:,1]):.2f}px | 视差 med={np.median(st[:,2]):.0f}px")

    # 片段级接受：候选中 |dy| 中位数 ≤ ACCEPT_MED_DY 的全部采用。
    # 同一粒子被跟踪器断成多段时，每个片段对都给出独立的 3D 时空采样——
    # 对波场拟合而言是更多数据而非冗余（重复点随后按帧去重）。
    pairs = sorted(candidates, key=lambda p: p[2][1])
    pairs = [p for p in pairs if p[2][1] <= ACCEPT_MED_DY]
    print(f"片段级接受（|dy|中位 ≤ {ACCEPT_MED_DY}px）：{len(pairs)} 对")
    if pairs:
        st = np.array([p[2] for p in pairs])
        print(f"  [片段级] 共同帧 med={int(np.median(st[:,0]))} | "
              f"|dy|中位 med={np.median(st[:,1]):.2f}px | 视差 med={np.median(st[:,2]):.0f}px")

    trajs_3d = triangulate_pairs(pairs, rect_left, rect_right,
                                 raw_left, raw_right, P1, P2)

    # 质量过滤：深度范围 + 水平跨度
    kept = []
    for tr in trajs_3d:
        z_med = np.median(tr[:, 3])
        extent = max(np.ptp(tr[:, 1]), np.ptp(tr[:, 2])
                   )
        if DEPTH_RANGE[0] <= z_med <= DEPTH_RANGE[1] and extent <= MAX_EXTENT_UV:
            kept.append(tr)
    print(f"三角化后质量过滤：{len(kept)}/{len(trajs_3d)} 条保留")
    # 注意：保持按片段分组的输出（不在此去重合并）——系统深度偏差是逐片段的，
    # 下游 run_real_pinn 的逐轨迹 debias 必须在片段粒度上做；
    # 跨片段的重复点由下游在 debias 之后按 (帧,位置) 去重。
    if kept:
        allpts = np.vstack(kept)
        print(f"  总点数 {len(allpts)}（含跨片段重复），帧范围 "
              f"{int(allpts[:,0].min())}..{int(allpts[:,0].max())}")
        print(f"  X [{allpts[:,1].min():.0f}, {allpts[:,1].max():.0f}] mm, "
              f"Y [{allpts[:,2].min():.0f}, {allpts[:,2].max():.0f}] mm, "
              f"Z med {np.median(allpts[:,3]):.0f} mm")

    with open(out_pkl, "wb") as f:
        pickle.dump(kept, f)
    print(f"已保存 {out_pkl}")

    # 诊断图：匹配对的矫正轨迹 dy/disp 随帧变化 + 3D 散点
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5))
    ax0 = fig.add_subplot(131)
    for k, (iL, jR, _s) in enumerate(pairs[:20]):
        fl, pl = rect_left[iL]
        fr, pr = rect_right[jR]
        common = sorted(set(fl) & set(fr))
        dy = [pl[fl.index(f), 1] - pr[fr.index(f), 1] for f in common]
        ax0.plot(common, dy, lw=0.8)
    ax0.set_title("rectified dy vs frame (first 20 pairs)")
    ax0.set_xlabel("frame"); ax0.set_ylabel("dy (px)")
    ax1 = fig.add_subplot(132)
    for k, (iL, jR, _s) in enumerate(pairs[:20]):
        fl, pl = rect_left[iL]
        fr, pr = rect_right[jR]
        common = sorted(set(fl) & set(fr))
        dp = [pl[fl.index(f), 0] - pr[fr.index(f), 0] for f in common]
        ax1.plot(common, dp, lw=0.8)
    ax1.set_title("disparity vs frame")
    ax1.set_xlabel("frame"); ax1.set_ylabel("disp (px)")
    ax2 = fig.add_subplot(133, projection="3d")
    for tr in kept:
        ax2.plot(tr[:, 1], tr[:, 2], tr[:, 3], lw=0.6)
    ax2.set_title(f"3D trajectories (n={len(kept)})")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"已保存 {out_png}")
