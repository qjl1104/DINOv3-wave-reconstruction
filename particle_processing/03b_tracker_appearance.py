# particle_processing/03b_tracker_appearance.py
"""
外观辅助重捕跟踪器（v3 高密度检测配套）。

机理依据（2026-07-29 系列实验）：
  - 连续帧粒子运动 p99 ≈ 2.3px → 紧距离门 12px 即可身份极纯关联；
  - 断档重捕时 EKF 预测误差 ~40px，而泡沫间距 p25 ≈ 21px(v2)/~15px(v3)，
    纯距离门没有可行窗口（dt80 碎裂、dt40/20 崩坏已实证）；
  - 解法：重捕不看距离看内容——候选检测与轨迹最后观测的 21×21 patch 做
    NCC，≥NCC_THR 才允许重捕；同相机时序外观稳定（仿射问题只在跨相机侧）。

流程（每帧）：
  1. EKF(匀速) 预测全部活跃轨迹
  2. 一级关联：无断档轨迹 × 检测，KDTree 稀疏候选(≤GATE_NEAR) + 稀疏匈牙利
  3. 二级重捕：有断档轨迹 × 剩余检测，先按 GATE_FAR 粗筛，再 NCC 验证，
     以 (1-ncc) 为成本做稀疏匈牙利
  4. 未匹配检测新生轨迹（与活跃轨迹当前位置去重 DEDUP_PX）
  5. 断档计数 +max_age 淘汰；成功匹配则刷新模板 patch

输出与 03 生产格式兼容：rr.Track 列表（.id, .points={frame: (x,y)}），
可直接接 clean_tracks_jumpcut.py → rematch_dino_v2.py → eval_tracks.py。

用法：
  ../.venv_fs/Scripts/python.exe 03b_tracker_appearance.py det_l det_r out_l out_r
      [--max-frames N] [--max-age 8] [--gate-near 12] [--gate-far 50] [--ncc-thr 0.55]
"""
import argparse
import glob
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rematch_rectified as rr  # noqa: E402

CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")
PATCH = 21           # 模板 patch 边长（泡沫 ~16px）
DEDUP_PX = 6.0       # 新生轨迹与活跃轨迹去重半径
Q_PROC = 2.0         # 过程噪声(速度随机游走) px/帧
R_MEAS = 1.0         # 观测噪声 px


class Tracker:
    def __init__(self, tid, pos, frame, template):
        self.id = tid
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])  # x,y,vx,vy
        self.P = np.diag([1.0, 1.0, 25.0, 25.0])
        self.points = {frame: (float(pos[0]), float(pos[1]))}
        self.last_seen = frame
        self.misses = 0
        self.template = template  # (PATCH,PATCH) float32 或 None

    def predict(self):
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        Q = np.diag([0.25, 0.25, Q_PROC, Q_PROC])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:2]

    def update(self, pos):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
        y = np.array(pos) - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * R_MEAS
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


def extract_patch(img_f32, x, y, ps=PATCH):
    """亚像素采样 patch（边界外返回 None）。img_f32 须为 float32 图（每帧只转一次）。"""
    h, w = img_f32.shape
    r = ps // 2
    if x - r < 0 or y - r < 0 or x + r + 1 > w or y + r + 1 > h:
        return None
    xs, ys = np.meshgrid(np.arange(ps) - r + x, np.arange(ps) - r + y)
    return map_coordinates(img_f32, [ys, xs], order=1, mode="nearest")


def ncc(a, b):
    if a is None or b is None:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return -1.0
    return float((a * b).sum() / (na * nb))


def sparse_match(tracks, dets, gate, extra_cost=None):
    """KDTree 稀疏候选 + 稀疏二部匹配（每行带虚拟列，未匹配者落虚拟列）。
    返回 matches[(ti,di)], 未匹配轨迹, 未匹配检测。"""
    if not tracks:
        return [], set(), set(range(len(dets)))
    if not dets:
        return [], set(range(len(tracks))), set()
    preds = np.array([t.x[:2] for t in tracks])
    darr = np.array(dets)
    tree = cKDTree(darr)
    pairs = tree.query_ball_point(preds, r=gate)
    rows, cols, vals = [], [], []
    for ti, cand in enumerate(pairs):
        for di in cand:
            dist = np.linalg.norm(preds[ti] - darr[di])
            c = dist
            if extra_cost is not None:
                ec = extra_cost(ti, di)
                if ec is None:
                    continue
                c = ec
            rows.append(ti)
            cols.append(di)
            vals.append(c)
    n_t, n_d = len(tracks), len(dets)
    BIG = 1e6
    # 每行一个专属虚拟列（成本 BIG）保证存在完全匹配
    rows += list(range(n_t))
    cols += [n_d + i for i in range(n_t)]
    vals += [BIG] * n_t
    mat = csr_matrix((vals, (rows, cols)), shape=(n_t, n_d + n_t))
    ri, ci = min_weight_full_bipartite_matching(mat)
    matches = [(t, d) for t, d in zip(ri, ci) if d < n_d]
    mt, md = set(t for t, _ in matches), set(d for _, d in matches)
    return matches, set(range(n_t)) - mt, set(range(n_d)) - md


def track_side(side, det_pkl, out_pkl, args):
    with open(det_pkl, "rb") as f:
        detections = pickle.load(f)
    # --no-foam：剔除来自大连通域（天然水沫团）的检测点（v3 meta big_comp=1）。
    # 依据：标识物 30mm 圆片尺寸已知，>1500px² 连通域不可能是标识物/小粘连对；
    # 天然水沫变形、质心漂移，不是可靠的拉格朗日标记。
    if getattr(args, "no_foam", False):
        meta_pkl = det_pkl.replace(".pkl", "_meta.pkl")
        with open(meta_pkl, "rb") as f:
            metas = pickle.load(f)
        n0 = sum(len(d) for d in detections)
        detections = [[p for p, m in zip(d, mm) if len(m) < 4 or m[3] < 1]
                      for d, mm in zip(detections, metas)]
        n1 = sum(len(d) for d in detections)
        print(f"  {side}: 泡沫过滤 {n0} → {n1} 点（剔除 {n0 - n1}，{(1 - n1 / max(n0, 1)) * 100:.1f}%）")
    calib = np.load(CALIB)
    map1, map2 = calib[f"map1_{side}"], calib[f"map2_{side}"]
    img_files = sorted(glob.glob(os.path.join(ROOT, f"data/{side}_images/*.bmp")))
    n_frames = min(len(detections), args.max_frames or len(detections))

    tracks, done, next_id = [], [], 0
    for fi in range(n_frames):
        dets = [tuple(map(float, p)) for p in detections[fi]]
        raw = cv2.imread(img_files[fi], 0)
        cur_img = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR).astype(np.float32) if raw is not None else None

        for t in tracks:
            t.predict()

        # 一级：无断档轨迹，紧门
        gapless = [i for i, t in enumerate(tracks) if t.misses == 0]
        sub = [tracks[i] for i in gapless]
        m1, _, _ = sparse_match(sub, dets, args.gate_near)
        used_d = set(d for _, d in m1)  # 已被匹配的检测
        matched_t = set()
        for si, di in m1:
            t = sub[si]
            t.update(dets[di])
            t.points[fi] = dets[di]
            t.last_seen = fi
            t.template = extract_patch(cur_img, *dets[di]) if cur_img is not None else t.template
            matched_t.add(gapless[si])
            used_d.add(di)

        # 二级：断档轨迹重捕，NCC 验证
        gapped = [i for i, t in enumerate(tracks) if t.misses > 0 and i not in matched_t]
        if gapped and cur_img is not None:
            sub = [tracks[i] for i in gapped]
            dets_rem = [d for j, d in enumerate(dets) if j not in used_d]
            rem_idx = [j for j in range(len(dets)) if j not in used_d]

            def cost(ti, di, _sub=sub, _dets=dets_rem):
                d = np.linalg.norm(_sub[ti].x[:2] - _dets[di])
                if d > args.gate_far:
                    return None
                s = ncc(_sub[ti].template, extract_patch(cur_img, *_dets[di]))
                if s < args.ncc_thr:
                    return None
                return 1.0 - s

            m2, um2, used2 = sparse_match(sub, dets_rem, args.gate_far, extra_cost=cost)
            for si, di in m2:
                t = sub[si]
                pos = dets_rem[di]
                t.update(pos)
                t.points[fi] = pos
                t.last_seen = fi
                t.misses = 0
                t.template = extract_patch(cur_img, *pos)
                matched_t.add(gapped[si])
                used_d.add(rem_idx[di])

        # 淘汰与留存
        survivors = []
        for i, t in enumerate(tracks):
            if i in matched_t:
                t.misses = 0
                survivors.append(t)
            else:
                t.misses += 1
                if t.misses <= args.max_age:
                    survivors.append(t)
                else:
                    done.append(t)
        tracks = survivors

        # 新生轨迹（去重）
        cur_pos = np.array([t.x[:2] for t in tracks]) if tracks else np.zeros((0, 2))
        for j, d in enumerate(dets):
            if j in used_d:
                continue
            if len(cur_pos) and np.min(np.linalg.norm(cur_pos - d, axis=1)) < DEDUP_PX:
                continue
            tracks.append(Tracker(next_id, d, fi, extract_patch(cur_img, *d) if cur_img is not None else None))
            next_id += 1
            cur_pos = np.array([t.x[:2] for t in tracks])

        if (fi + 1) % 200 == 0:
            print(f"  {side} {fi + 1}/{n_frames}: 活跃 {len(tracks)} 完成 {len(done)}", flush=True)

    done.extend(tracks)
    out = []
    for t in done:
        nt = rr.Track(track_id=t.id)
        nt.points = t.points
        out.append(nt)
    lens = np.array([len(t.points) for t in out]) if out else np.array([0])
    print(f"{side}: 轨迹 {len(out)} 条 | 长≥20: {(lens >= 20).sum()} | "
          f"长≥100: {(lens >= 100).sum()} | 中位长 {np.median(lens):.0f}")
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"saved {out_pkl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("det_l")
    ap.add_argument("det_r")
    ap.add_argument("out_l")
    ap.add_argument("out_r")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-age", type=int, default=8)
    ap.add_argument("--gate-near", type=float, default=12.0)
    ap.add_argument("--gate-far", type=float, default=50.0)
    ap.add_argument("--ncc-thr", type=float, default=0.55)
    ap.add_argument("--no-foam", action="store_true",
                    help="剔除 v3 meta 中 big_comp=1 的天然水沫检测点")
    args = ap.parse_args()
    track_side("left", args.det_l, args.out_l, args)
    track_side("right", args.det_r, args.out_r, args)


if __name__ == "__main__":
    main()
