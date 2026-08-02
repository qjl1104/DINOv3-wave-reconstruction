# wave_modeling/bias_field.py
# -*- coding: utf-8 -*-
"""
静态偏差场 B(x,y) 估计与校正（替代逐片段中位数 debias 的物理方案）。

机理：每条轨迹的常值深度偏移（std ~20-30mm）疑为标定残差的空间系统误差。
由于波面时均值为零，长轨迹（≥200 帧 ≈ 2.5+ 周期）的时间中位数 ≈ B(x,y)。
用长轨迹估计平滑偏差场，再从全部点中扣除——不抹掉各片段的真实 DC，
比逐片段中位数去偏（强制每段均值归零）物理上更诚实。

用法：.venv_fs/Scripts/python.exe wave_modeling/bias_field.py [traj_pkl]
输出：data/trajectories/trajectories_3d_v3nf_hung_biascorr.pkl + 诊断打印 + bias_field.png
"""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 50.0
MIN_LONG = 200  # 估计偏差场用的最短轨迹（帧）


def fit_plane(pts):
    c = pts.mean(0)
    cov = np.cov((pts - c).T)  # 3x3 协方差，避免对 (N,3) 直接 SVD 的内存爆炸
    _, vt = np.linalg.eigh(cov)
    n = vt[:, 0] * np.sign(vt[2, 0])  # 最小特征值对应法向
    return c, n


def main():
    pkl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "data/trajectories/trajectories_3d_v3nf_hung_dino.pkl")
    with open(pkl, "rb") as f:
        trajs = pickle.load(f)
    pts_all = np.vstack([t[:, 1:4] for t in trajs if len(t)])
    c0, n = fit_plane(pts_all)

    # 每条轨迹: 中位位置, η 时间中位数
    recs = []
    for t in trajs:
        if len(t) < 10:
            continue
        eta = (t[:, 1:4] - c0) @ n
        pos = np.median(t[:, 1:4], axis=0)
        recs.append((pos[0], pos[1], np.median(eta), len(t)))
    recs = np.array(recs)
    long = recs[recs[:, 3] >= MIN_LONG]
    print(f"片段 {len(recs)} 条, 用于估计偏差场的长片段(≥{MIN_LONG}帧) {len(long)} 条")

    # 方案 A: 常数; 方案 B: 平面; 方案 C: 分箱中位网格(250mm) 局部平滑
    def eval_bias(bias_of):
        resid = recs[:, 2] - bias_of(recs[:, 0], recs[:, 1])
        return resid.std()
    bias_const = np.median(long[:, 2])
    A = np.column_stack([long[:, 0], long[:, 1], np.ones(len(long))])
    coef, *_ = np.linalg.lstsq(A, long[:, 2], rcond=None)
    bias_plane = lambda x, y: coef[0] * x + coef[1] * y + coef[2]

    # 分箱中位 + 高斯平滑
    from scipy.ndimage import gaussian_filter
    bx = np.arange(recs[:, 0].min(), recs[:, 0].max() + 250, 250)
    by = np.arange(recs[:, 1].min(), recs[:, 1].max() + 250, 250)
    Hsum = np.zeros((len(bx) - 1, len(by) - 1))
    Hcnt = np.zeros_like(Hsum)
    Hmed = np.zeros_like(Hsum)
    idx_x = np.clip(np.digitize(long[:, 0], bx) - 1, 0, Hsum.shape[0] - 1)
    idx_y = np.clip(np.digitize(long[:, 1], by) - 1, 0, Hsum.shape[1] - 1)
    for i in range(len(long)):
        Hsum[idx_x[i], idx_y[i]] += long[i, 2]
        Hcnt[idx_x[i], idx_y[i]] += 1
    for i in range(Hsum.shape[0]):
        for j in range(Hsum.shape[1]):
            sel = long[(idx_x == i) & (idx_y == j), 2]
            Hmed[i, j] = np.median(sel) if len(sel) else np.nan
    Hfill = np.where(Hcnt > 0, Hmed, bias_const)
    Hsmooth = gaussian_filter(Hfill, 1.0)

    def bias_grid(x, y):
        ix = np.clip(np.digitize(x, bx) - 1, 0, Hsmooth.shape[0] - 1)
        iy = np.clip(np.digitize(y, by) - 1, 0, Hsmooth.shape[1] - 1)
        return Hsmooth[ix, iy]

    print(f"偏差场诊断: 长片段 η 中位数 std = {long[:, 2].std():.2f} mm")
    print(f"  常数模型: 残差 std = {eval_bias(lambda x, y: bias_const):.2f} mm")
    print(f"  平面模型: 残差 std = {eval_bias(bias_plane):.2f} mm "
          f"(梯度 {coef[0]*1e3:.2f}/{coef[1]*1e3:.2f} mm/m)")
    print(f"  网格模型: 残差 std = {eval_bias(bias_grid):.2f} mm")

    # 网格模型若明显优于平面才用之(防过拟合), 否则用平面
    use_grid = eval_bias(bias_grid) < eval_bias(bias_plane) - 1.0
    bias_fn = bias_grid if use_grid else bias_plane
    print(f"  采用: {'网格' if use_grid else '平面'}")

    # 应用校正 → 新 pkl：η 减 B(x,y)，等价于把点沿平面法向移动 -B
    out_trajs = []
    bias_before, bias_after = [], []
    for t in trajs:
        if len(t) < 10:
            continue
        eta = (t[:, 1:4] - c0) @ n
        B = bias_fn(t[:, 1], t[:, 2])
        eta_corr = eta - B
        t2 = t.copy()
        t2[:, 1:4] = t[:, 1:4] - B[:, None] * n[None, :]
        bias_before.append(np.median(eta))
        bias_after.append(np.median(eta_corr))
        out_trajs.append(t2)
    print(f"轨迹间偏移 std: {np.std(bias_before):.2f} → {np.std(bias_after):.2f} mm")
    eta_all_before = np.concatenate([(t[:, 1:4] - c0) @ n for t in trajs if len(t)])
    eta_all_after = np.concatenate([(t[:, 1:4] - c0) @ n for t in out_trajs if len(t)])
    print(f"η std: {eta_all_before.std():.2f} → {eta_all_after.std():.2f} mm")

    out_pkl = os.path.join(ROOT, "data/trajectories/trajectories_3d_v3nf_hung_biascorr.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(out_trajs, f)
    np.savez(os.path.join(ROOT, "wave_modeling/real_run/bias_field.npz"),
             bx=bx, by=by, Hsmooth=Hsmooth, coef=coef, use_grid=use_grid)
    print(f"saved {out_pkl}")

    # 诊断图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    im = ax[0].pcolormesh(by / 1000, bx / 1000, Hsmooth)
    ax[0].set_title("B(x,y) mm (250mm grid, smoothed)")
    ax[0].set_xlabel("Y m"); ax[0].set_ylabel("X m")
    fig.colorbar(im, ax=ax[0])
    ax[1].hist(bias_before, bins=40, alpha=0.6, label="before")
    ax[1].hist(bias_after, bins=40, alpha=0.6, label="after")
    ax[1].set_title("inter-track bias"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "wave_modeling/real_run/bias_field.png"), dpi=110)
    print("saved bias_field.png")


if __name__ == "__main__":
    main()
