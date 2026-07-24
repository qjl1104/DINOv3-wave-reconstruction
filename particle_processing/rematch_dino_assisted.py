# particle_processing/rematch_dino_assisted.py
"""
【已归档】本脚本是 aug24 对照侧链（repro 链），非生产链——生产链为
rematch_dino_v2.py（canonical 输入）。其描述子输入 DINOv3/desc_aug24_*.pkl
是历史预存产物，本分支上没有对应的再生成脚本，丢失后无法在本分支重建。

DINOv3 辅助跨相机轨迹匹配：在 rematch_rectified 的几何判据（dy≈0、视差近恒定）
之上，加"轨迹级描述子相似度"作为消歧打分/门槛。

背景：aug24 链（repro）2D 轨迹不少（75+106 条、中位 183/250 帧）但几何判据
只产出 6 条 3D 片段——大量真匹配被视差波动门（std≤30）误杀，而错误匹配
又无法区分。DINOv3 语义描述子提供与几何无关的独立证据：
对候选对，取共同帧上左右轨迹点（映射回本帧最近检测点，≤3px）的
描述子余弦相似度均值。真匹配的 DINO 相似度应显著高于错配。

用法（缺省对照 repro 链）：
  ../.venv_fs/Scripts/python.exe rematch_dino_assisted.py
输出：../data/trajectories/trajectories_3d_repro_dino.pkl + 打印统计
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

TRAJ_L = os.path.join(ROOT, "data/trajectories/trajectories_2d_left_repro.pkl")
TRAJ_R = os.path.join(ROOT, "data/trajectories/trajectories_2d_right_repro.pkl")
DET_L = os.path.join(ROOT, "data/detections/detections_left.pkl")
DET_R = os.path.join(ROOT, "data/detections/detections_right.pkl")
DESC_L = os.path.join(ROOT, "DINOv3/desc_aug24_left.pkl")
DESC_R = os.path.join(ROOT, "DINOv3/desc_aug24_right.pkl")
CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")
OUT_PKL = os.path.join(ROOT, "data/trajectories/trajectories_3d_repro_dino.pkl")

MAX_NN_DIST = 3.0      # 轨迹点 → 检测点归属上限 px
# 视差平面先验开关在 rematch_rectified（rr.DISP_PLANE / rr.DISP_PLANE_TOL，默认开启，
# 系数已交叉验证）：消长峰波沿波峰（≈极线）方向的匹配欠约束；重拟合用 fit_disp_plane.py。


def load_det_desc(det_pkl, desc_pkl):
    dets = pickle.load(open(det_pkl, "rb"))
    descs = pickle.load(open(desc_pkl, "rb"))
    out = []
    for d, s in zip(dets, descs):
        pos = np.array(d).reshape(-1, 2) if len(d) else np.zeros((0, 2))
        out.append((pos, s.astype(np.float32)))
    return out


def track_descriptors(track, det_desc):
    """轨迹 → {frame: 768-d 描述子}（轨迹点映射到本帧最近检测点）。"""
    res = {}
    for f, (x, y) in track.points.items():
        if f >= len(det_desc):
            continue
        pos, desc = det_desc[f]
        if len(pos) == 0:
            continue
        d = np.linalg.norm(pos - np.array([x, y]), axis=1)
        j = int(np.argmin(d))
        if d[j] <= MAX_NN_DIST:
            res[f] = desc[j]
    return res


def main():
    import cv2
    raw_left = pickle.load(open(TRAJ_L, "rb"))
    raw_right = pickle.load(open(TRAJ_R, "rb"))
    calib = np.load(CALIB)
    KL, DL = calib["K_left"], calib["D_left"].ravel()
    KR, DR = calib["K_right"], calib["D_right"].ravel()
    R1, R2, P1, P2 = calib["R1"], calib["R2"], calib["P1"], calib["P2"]
    # rematch_rectified.triangulate_pairs 引用其模块级 KL/DL/KR/DR 与 R1/R2
    # （原本在 __main__ 块赋值，import 时不存在）——在此补齐
    rr.KL, rr.DL, rr.KR, rr.DR = KL, DL, KR, DR
    rr.R1, rr.R2 = R1, R2
    raw_left = [t for t in raw_left if len(t.points) >= rr.MIN_TRAJ_LEN]
    raw_right = [t for t in raw_right if len(t.points) >= rr.MIN_TRAJ_LEN]
    print(f"参与匹配：左 {len(raw_left)} 条，右 {len(raw_right)} 条")
    rect_left = [rr.rectify_traj(t, KL, DL, R1, P1) for t in raw_left]
    rect_right = [rr.rectify_traj(t, KR, DR, R2, P2) for t in raw_right]

    hungarian, candidates = rr.match_pairs(rect_left, rect_right)
    print(f"[几何] 匈牙利 {len(hungarian)} 对；硬过滤候选 {len(candidates)} 对")

    det_desc_l = load_det_desc(DET_L, DESC_L)
    det_desc_r = load_det_desc(DET_R, DESC_R)
    print("[DINO] 计算轨迹描述子…")
    td_l = [track_descriptors(t, det_desc_l) for t in raw_left]
    td_r = [track_descriptors(t, det_desc_r) for t in raw_right]

    # 给所有几何候选对打 DINO 分；同时构造"放宽视差波动门"的扩展候选
    rows = dc.score_candidates(candidates, rect_left, rect_right, td_l, td_r)
    sims = np.array([r["sim"] for r in rows if r["nsim"] >= dc.MIN_NSIM])
    if len(sims):
        print(f"[DINO] 候选对相似度分布：med {np.median(sims):.3f} "
              f"p25 {np.percentile(sims, 25):.3f} p75 {np.percentile(sims, 75):.3f}")

    ext = dc.extended_candidates(rect_left, rect_right, td_l, td_r)
    print(f"[放宽] 视差波动 {rr.MAX_DISP_STD:.0f}–{dc.RELAX_DISP_STD:.0f}px 的扩展候选 {len(ext)} 对")

    # 最终接受：几何候选中 DINO ≥ 门 + 扩展候选中 DINO ≥ 门
    acc, _ = dc.dino_accept(rows)
    acc_ext, _ = dc.dino_accept(ext)
    print(f"[接受] 几何候选×DINO≥{dc.DINO_GATE}: {len(acc)} 对；扩展×DINO: {len(acc_ext)} 对")
    pairs = [(r["i"], r["j"], r["st"]) for r in acc + acc_ext]

    trajs_3d = rr.triangulate_pairs(pairs, rect_left, rect_right,
                                    raw_left, raw_right, P1, P2)
    kept = dc.quality_filter(trajs_3d)
    print(f"三角化后质量过滤：{len(kept)}/{len(trajs_3d)} 条保留，"
          f"总点 {sum(len(t) for t in kept)}")
    with open(OUT_PKL, "wb") as f:
        pickle.dump(kept, f)
    print(f"[输出] {OUT_PKL}")


if __name__ == "__main__":
    main()
