# DINOv3/track_descriptor.py
"""
实验A：DINOv3 描述子时序跟踪（替代 KLT），目标：更长更连续的粒子轨迹。

原理：时序重识别的外观变化远小于双目跨相机，正是冻结基础模型特征的强项。
每帧从 feature_cache 取 blob 关键点 + 双线性采样描述子，帧间用
"空间门控（≤GATE_PX）+ 描述子余弦相似度（≥SIM_MIN）"的匈牙利指派做关联，
允许 max_age 帧断检重识别（描述子跨过短时遮挡/漏检，这是比纯 KLT
位置预测强的关键点）。

输出（供 match_dino_tracks.py 立体匹配）：
    data_dino_tracks_left.pkl / data_dino_tracks_right.pkl
    list[Track]：Track.points = {frame: (x, y)}，矫正图像像素坐标
    （paper_params_recalculated.npz 的矫正坐标系，与 feature_cache 同源）

用法：../.venv_fs/Scripts/python.exe track_descriptor.py
"""

import glob
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_descriptor_nn import sample_desc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GATE_PX = 8.0       # 连续帧空间门控 px（粒子表观运动 ~1-2px/帧；25px 会在
                    # ~330 个相似 blob/帧 中造成身份漂移——实测 60% 长轨迹含 >8px 跳变）
GATE_PER_MISS = 4.0  # 每多断 1 帧额外放宽 px（断档重识别时位置不确定性增长）
GATE_MAX = 30.0
SIM_MIN = 0.65      # 描述子余弦相似度下限
MAX_AGE = 10        # 允许断检帧数（描述子重识别）
MIN_HITS = 15       # 最短轨迹（帧数）


class Track:
    """与 rematch_rectified 兼容的最小轨迹对象。"""

    def __init__(self, tid, frame, pt, desc):
        self.id = tid
        self.points = {frame: tuple(pt)}
        self.last_desc = desc
        self.last_frame = frame
        self.misses = 0

    @property
    def pos(self):
        return np.array(self.points[self.last_frame])


def track_camera(side, files):
    tracks, next_id = [], 0
    active = []
    for fi, fp in enumerate(files):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        kps = d[f"keypoints_{side}"].float()
        descs = sample_desc(d[f"feat_{side}"], kps).numpy()
        pts = kps.numpy()

        if active:
            P = np.array([t.pos for t in active])
            D = np.stack([t.last_desc for t in active])
            sim = D @ descs.T                                # [T, N]
            dist = np.linalg.norm(P[:, None, :] - pts[None, :, :], axis=2)
            # 门控随断档帧数放宽：连续帧 8px，每多断 1 帧 +4px，封顶 30px
            gap = np.array([fi - t.last_frame for t in active])
            gate = np.minimum(GATE_PX + GATE_PER_MISS * (gap - 1), GATE_MAX)
            cost = 1.0 - sim
            cost[dist > gate[:, None]] = np.inf
            cost[sim < SIM_MIN] = np.inf
            BIG = 1e4
            ok_r = np.isfinite(cost).any(axis=1)
            ok_c = np.isfinite(cost).any(axis=0)
            matched_t, matched_d = set(), set()
            if ok_r.any() and ok_c.any():
                sub = np.where(np.isfinite(cost), cost, BIG)[np.ix_(ok_r, ok_c)]
                rr, cc = linear_sum_assignment(sub)
                ridx, cidx = np.where(ok_r)[0], np.where(ok_c)[0]
                for r, c in zip(rr, cc):
                    if sub[r, c] < BIG:
                        ti, di = ridx[r], cidx[c]
                        t = active[ti]
                        t.points[fi] = tuple(pts[di])
                        t.last_desc = descs[di]
                        t.last_frame = fi
                        t.misses = 0
                        matched_t.add(ti)
                        matched_d.add(di)
            for ti, t in enumerate(active):
                if ti not in matched_t:
                    t.misses += 1
            active = [t for ti, t in enumerate(active) if t.misses <= MAX_AGE]
            unmatched = [di for di in range(len(pts)) if di not in matched_d]
        else:
            unmatched = list(range(len(pts)))

        for di in unmatched:
            tr = Track(next_id, fi, pts[di], descs[di])
            next_id += 1
            tracks.append(tr)
            active.append(tr)

        if (fi + 1) % 200 == 0:
            print(f"  [{side}] {fi + 1}/{len(files)} 活跃 {len(active)}")

    long_tracks = [t for t in tracks if len(t.points) >= MIN_HITS]
    lens = np.array([len(t.points) for t in tracks])
    lens_l = np.array([len(t.points) for t in long_tracks])
    print(f"[{side}] 总轨迹 {len(tracks)}，≥{MIN_HITS} 帧 {len(long_tracks)} 条 | "
          f"长轨迹长度 中位 {np.median(lens_l) if len(lens_l) else 0:.0f} "
          f"p90 {np.percentile(lens_l, 90) if len(lens_l) else 0:.0f} "
          f"max {lens_l.max() if len(lens_l) else 0}")
    return long_tracks


def main():
    files = sorted(glob.glob(os.path.join(HERE, "feature_cache/left*.pt")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    print(f"缓存帧数: {len(files)}")
    t0 = time.time()
    for side in ["left", "right"]:
        tracks = track_camera(side, files)
        out = os.path.join(HERE, f"data_dino_tracks_{side}.pkl")
        with open(out, "wb") as f:
            pickle.dump(tracks, f)
        print(f"[输出] {out}（{time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
