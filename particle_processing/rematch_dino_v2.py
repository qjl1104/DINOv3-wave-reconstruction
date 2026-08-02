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
import rematch_dino_common as dc  # noqa: E402
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
# 视差平面先验开关在 rematch_rectified（rr.DISP_PLANE / rr.DISP_PLANE_TOL，默认开启，
# 系数已交叉验证）：消长峰波沿波峰（≈极线）方向的匹配欠约束；重拟合用 fit_disp_plane.py。


def main():
    # 可选路径覆盖: rematch_dino_v2.py [traj_l traj_r desc_l desc_r out_pkl] [--hung-only]
    # --hung-only: 只做匈牙利一对一匹配（最严口径，生产 canonical 用）
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    hung_only = "--hung-only" in sys.argv
    traj_l = pos[0] if len(pos) > 0 else rr.TRAJ_L
    traj_r = pos[1] if len(pos) > 1 else rr.TRAJ_R
    desc_l = pos[2] if len(pos) > 2 else os.path.join(ROOT, "DINOv3/desc_v2tracks_left.pkl")
    desc_r = pos[3] if len(pos) > 3 else os.path.join(ROOT, "DINOv3/desc_v2tracks_right.pkl")
    out_pkl = pos[4] if len(pos) > 4 else OUT_PKL
    raw_left = pickle.load(open(traj_l, "rb"))
    raw_right = pickle.load(open(traj_r, "rb"))
    td_l = pickle.load(open(desc_l, "rb"))
    td_r = pickle.load(open(desc_r, "rb"))
    calib = np.load(CALIB)
    KL, DL = calib["K_left"], calib["D_left"].ravel()
    KR, DR = calib["K_right"], calib["D_right"].ravel()
    R1, R2, P1, P2 = calib["R1"], calib["R2"], calib["P1"], calib["P2"]
    rr.KL, rr.DL, rr.KR, rr.DR = KL, DL, KR, DR
    rr.R1, rr.R2 = R1, R2  # triangulate_pairs 矫正三角化用
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

    if hung_only:
        # 匈牙利一对一模式：直接用匈牙利对，不做候选扩展/DINO 门
        pairs = [(i, j, st) for (i, j, st) in hungarian]
        print(f"[匈牙利一对一] {len(pairs)} 对直接三角化")
    else:
        rows = dc.score_candidates(candidates, rect_left, rect_right, td_l, td_r)
        sims = np.array([r["sim"] for r in rows if r["nsim"] >= dc.MIN_NSIM])
        if len(sims):
            print(f"[DINO] 几何候选相似度：med {np.median(sims):.3f} "
                  f"p25 {np.percentile(sims, 25):.3f} p75 {np.percentile(sims, 75):.3f} "
                  f"min {sims.min():.3f}")

        ext = dc.extended_candidates(rect_left, rect_right, td_l, td_r)
        print(f"[放宽] 视差波动 {rr.MAX_DISP_STD:.0f}–{dc.RELAX_DISP_STD:.0f}px 扩展候选 {len(ext)} 对")

        acc, rej = dc.dino_accept(rows)
        acc_ext, _ = dc.dino_accept(ext)
        print(f"[接受] 几何候选×DINO≥{dc.DINO_GATE}: {len(acc)}/{len(rows)} 对（拒 {len(rej)} 对）；"
              f"扩展×DINO: {len(acc_ext)}/{len(ext)} 对")
        if rej:
            print(f"  被拒对相似度：{[round(r['sim'], 2) for r in rej]}")
        pairs = [(r["i"], r["j"], r["st"]) for r in acc + acc_ext]

    trajs_3d = rr.triangulate_pairs(pairs, rect_left, rect_right,
                                    raw_left, raw_right, P1, P2)
    kept = dc.quality_filter(trajs_3d)
    print(f"三角化后质量过滤：{len(kept)}/{len(trajs_3d)} 条保留，"
          f"总点 {sum(len(t) for t in kept)}（基线 v2：46 条 / 9866 点）")
    with open(out_pkl, "wb") as f:
        pickle.dump(kept, f)
    print(f"[输出] {out_pkl}")


if __name__ == "__main__":
    main()
