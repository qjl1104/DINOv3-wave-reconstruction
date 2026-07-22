# particle_processing/rematch_dino_v2.py
"""
v2 轨迹 × DINOv3 辅助匹配（canonical 链上的 DINO 增强实验）。

与 rematch_rectified 的几何判据相比，加两层使用描述子：
1. 几何候选对（dy/视差门全过）打 DINO 分，作为真伪复核（低分可拒）；
2. 扩展候选：视差波动放宽到 60px 的对，DINO 高分者救回。
真匹配在 v2 上应给出 >46 条且保持相位相干的 3D 片段。

用法：../.venv_fs/Scripts/python.exe rematch_dino_v2.py
输出：../data/trajectories/trajectories_3d_v2_dino.pkl + 统计打印
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


class RobustKalmanFilter(rr.SimpleKalmanFilter):
    pass


__main__.RobustKalmanFilter = RobustKalmanFilter

CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")
OUT_PKL = os.path.join(ROOT, "data/trajectories/trajectories_3d_v2_dino.pkl")
DINO_GATE = 0.55
RELAX_DISP_STD = 60.0


def pair_dino_sim(dl, dr, common):
    sims = []
    for f in common:
        if f in dl and f in dr:
            a, b = dl[f], dr[f]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-6 and nb > 1e-6:
                sims.append(float(a @ b / (na * nb)))
    return (float(np.mean(sims)) if sims else -1.0), len(sims)


def main():
    raw_left = pickle.load(open(os.path.join(
        ROOT, "data/trajectories/trajectories_2d_left_optimized.pkl"), "rb"))
    raw_right = pickle.load(open(os.path.join(
        ROOT, "data/trajectories/trajectories_2d_right_optimized.pkl"), "rb"))
    td_l = pickle.load(open(os.path.join(ROOT, "DINOv3/desc_v2tracks_left.pkl"), "rb"))
    td_r = pickle.load(open(os.path.join(ROOT, "DINOv3/desc_v2tracks_right.pkl"), "rb"))
    calib = np.load(CALIB)
    KL, DL = calib["K_left"], calib["D_left"].ravel()
    KR, DR = calib["K_right"], calib["D_right"].ravel()
    R1, R2, P1, P2 = calib["R1"], calib["R2"], calib["P1"], calib["P2"]
    rr.KL, rr.DL, rr.KR, rr.DR = KL, DL, KR, DR
    # 轨迹与描述子按同一顺序联合过滤（MIN_TRAJ_LEN），保持索引对齐
    pl_ = [(t, d) for t, d in zip(raw_left, td_l) if len(t.points) >= rr.MIN_TRAJ_LEN]
    raw_left, td_l = [t for t, _ in pl_], [d for _, d in pl_]
    pr_ = [(t, d) for t, d in zip(raw_right, td_r) if len(t.points) >= rr.MIN_TRAJ_LEN]
    raw_right, td_r = [t for t, _ in pr_], [d for _, d in pr_]
    print(f"参与匹配：左 {len(raw_left)} 条，右 {len(raw_right)} 条")
    rect_left = [rr.rectify_traj(t, KL, DL, R1, P1) for t in raw_left]
    rect_right = [rr.rectify_traj(t, KR, DR, R2, P2) for t in raw_right]

    hungarian, candidates = rr.match_pairs(rect_left, rect_right)
    print(f"[几何] 匈牙利 {len(hungarian)} 对；硬过滤候选 {len(candidates)} 对")

    rows = []
    for i, j, st in candidates:
        fl, _ = rect_left[i]
        fr, _ = rect_right[j]
        common = sorted(set(fl) & set(fr))
        sim, nsim = pair_dino_sim(td_l[i], td_r[j], common)
        rows.append(dict(i=i, j=j, st=st, sim=sim, nsim=nsim))
    sims = np.array([r["sim"] for r in rows if r["nsim"] >= 10])
    if len(sims):
        print(f"[DINO] 几何候选相似度：med {np.median(sims):.3f} "
              f"p25 {np.percentile(sims, 25):.3f} p75 {np.percentile(sims, 75):.3f} "
              f"min {sims.min():.3f}")

    ext = []
    for i, t_l in enumerate(raw_left):
        fl, pl = rect_left[i]
        fset_l = set(fl)
        for j, t_r in enumerate(raw_right):
            fr, pr = rect_right[j]
            common = sorted(fset_l & set(fr))
            if len(common) < rr.MIN_OVERLAP:
                continue
            il = [fl.index(f) for f in common]
            ir = [fr.index(f) for f in common]
            dy = pl[il, 1] - pr[ir, 1]
            disp = pl[il, 0] - pr[ir, 0]
            med_dy, med_disp, std_disp = np.median(np.abs(dy)), np.median(disp), disp.std()
            if med_dy > rr.MAX_MED_DY:
                continue
            if not (rr.DISP_RANGE[0] <= med_disp <= rr.DISP_RANGE[1]):
                continue
            if std_disp <= rr.MAX_DISP_STD or std_disp > RELAX_DISP_STD:
                continue
            sim, nsim = pair_dino_sim(td_l[i], td_r[j], common)
            ext.append(dict(i=i, j=j, st=(len(common), med_dy, med_disp, std_disp),
                            sim=sim, nsim=nsim))
    print(f"[放宽] 视差波动 {rr.MAX_DISP_STD:.0f}–{RELAX_DISP_STD:.0f}px 扩展候选 {len(ext)} 对")

    acc = [r for r in rows if r["nsim"] >= 10 and r["sim"] >= DINO_GATE]
    acc_ext = [r for r in ext if r["nsim"] >= 10 and r["sim"] >= DINO_GATE]
    rej = [r for r in rows if r["nsim"] >= 10 and r["sim"] < DINO_GATE]
    print(f"[接受] 几何候选×DINO≥{DINO_GATE}: {len(acc)}/{len(rows)} 对（拒 {len(rej)} 对）；"
          f"扩展×DINO: {len(acc_ext)}/{len(ext)} 对")
    if rej:
        print(f"  被拒对相似度：{[round(r['sim'], 2) for r in rej]}")
    pairs = [(r["i"], r["j"], r["st"]) for r in acc + acc_ext]

    trajs_3d = rr.triangulate_pairs(pairs, rect_left, rect_right,
                                    raw_left, raw_right, P1, P2)
    kept = []
    for tr in trajs_3d:
        z_med = np.median(tr[:, 3])
        extent = max(np.ptp(tr[:, 1]), np.ptp(tr[:, 2]))
        if rr.DEPTH_RANGE[0] <= z_med <= rr.DEPTH_RANGE[1] and extent <= rr.MAX_EXTENT_UV:
            kept.append(tr)
    print(f"三角化后质量过滤：{len(kept)}/{len(trajs_3d)} 条保留，"
          f"总点 {sum(len(t) for t in kept)}（基线 v2：46 条 / 9866 点）")
    with open(OUT_PKL, "wb") as f:
        pickle.dump(kept, f)
    print(f"[输出] {OUT_PKL}")


if __name__ == "__main__":
    main()
