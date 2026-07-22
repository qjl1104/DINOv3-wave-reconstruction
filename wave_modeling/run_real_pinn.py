# wave_modeling/run_real_pinn.py
"""
真实数据端到端实验：data/trajectories/trajectories_3d_v2.pkl →
PCA 主平面坐标 → PINN v2 训练 → 留出集评估。

输入格式（rematch_rectified.py 的输出）：list[np.ndarray]，每条 (N,4)：
frame（绝对帧号）, X, Y, Z（mm，相机1坐标系）。

坐标处理：相机俯视视角下水面在相机坐标系中是倾斜平面，直接对 Z 取
残差会混入倾角放大（曾出现拟合平面"倾角 84.7°"的伪影）。改为 PCA：
SVD 求点云主平面，η = 沿法向的残差（真实的波面起伏），(u,v) 为面内坐标。

时间单位：pca_plane_coords 已按 FPS=50 把帧号换算为秒，c 单位即 mm/s。
c 固定为理论值 1976 mm/s（反演不可靠；数据独立测速见
diag_hovmoller_xcorr.py 的互谱相位法：1975 mm/s，95% CI [1867,2107]）。

用法：.venv_fs/Scripts/python.exe wave_modeling/run_real_pinn.py
"""

import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_v2 import PINNWaveV2, train_pinn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(ROOT, "data/trajectories/trajectories_3d_v2.pkl")
OUT = os.path.join(ROOT, "wave_modeling/real_run")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 50.0          # 采集帧率（刘晔恒论文：50 Hz）
C_THEORY = 1976.0   # 理论波速 mm/s（0.79 Hz 深水规则波，λ=2.5017m）


def load_trajectories(pkl_path, min_len=20):
    """v2 格式：每条 (N,4) = [frame, X, Y, Z]。"""
    with open(pkl_path, "rb") as f:
        trajs = pickle.load(f)
    return [np.asarray(t, dtype=np.float64) for t in trajs if len(t) >= min_len]


def pca_plane_coords(trajs):
    """PCA 拟合平均水面：返回每条的 (u, v, eta, t)，eta 为法向残差（mm），
    t 由帧号换算为秒。"""
    allpts = np.vstack([t[:, 1:4] for t in trajs])
    c0 = allpts.mean(axis=0)
    _, _, vt = np.linalg.svd(allpts - c0, full_matrices=False)
    e_u, e_v, n = vt[0], vt[1], vt[2]  # 法向 n 是最小奇异值方向
    series = []
    for t in trajs:
        d = t[:, 1:4] - c0
        series.append(np.c_[d @ e_u, d @ e_v, d @ n, t[:, 0] / FPS])
    return series, c0, n


def prepare_data(pkl_path=PKL, min_len=20, verbose=True):
    """加载轨迹 → PCA 主平面坐标 → MAD 法向离群剔除 → 逐片段 debias → 跨片段去重。
    返回 (series, allpts, meta)：series 为逐片段 [u,v,eta,t(s)]（未去重），
    allpts 为去重后的合并点云，meta = (plane_centroid, plane_normal)。
    供 main() 与诊断脚本（diag_hovmoller_xcorr.py）共用，保证预处理唯一出处。"""
    trajs = load_trajectories(pkl_path, min_len)
    series, c0, n = pca_plane_coords(trajs)
    allpts = np.vstack(series)
    n_traj = len(series)
    eta = allpts[:, 2]
    # 稳健剔除法向离群点（跟踪噪声导致的瞬间跳点）
    mad = 1.4826 * np.median(np.abs(eta - np.median(eta)))
    keep = np.abs(eta - np.median(eta)) < 5 * max(mad, 1e-6)
    series = [s[np.abs(s[:, 2] - np.median(eta)) < 5 * max(mad, 1e-6)] for s in series]
    allpts = np.vstack(series)
    eta = allpts[:, 2]
    if verbose:
        print(f"[数据] 轨迹 {n_traj} 条，点 {len(allpts)} 个（剔除法向离群 {np.sum(~keep)} 个）| "
              f"η std = {eta.std():.2f} mm, "
              f"范围 [{eta.min():.1f}, {eta.max():.1f}] mm")

    # 逐轨迹去均值（debias）：诊断显示轨迹间存在 ±300~560mm 的常值深度偏差，
    # 与 FoundationStereo 残差实验的"碗形"系统误差同源——标定/矫正残差导致的
    # 位置相关深度偏差。平均水面已知是平的，故每条轨迹的时间均值 ≈ 偏差本身，
    # 减去后只损失"平均水面为平"这一已知信息，保留波动的时间相位信息。
    # 注意：这是校正标定系统误差的工程手段；根治需重标定或静水参考帧。
    offsets = np.array([np.median(s[:, 2]) for s in series])
    series = [s.copy() for s in series]
    for s in series:
        s[:, 2] -= np.median(s[:, 2])
    allpts = np.vstack(series)
    eta = allpts[:, 2]
    if verbose:
        print(f"[debias] 轨迹间偏移 std = {offsets.std():.1f} mm（已逐轨迹减去中位数）| "
              f"去偏后 η std = {eta.std():.2f} mm, "
              f"范围 [{eta.min():.1f}, {eta.max():.1f}] mm")

    # 跨片段重复点去重（同一粒子同一帧经不同片段对重复三角化）：
    # 按 (帧, u/v 10mm 网格) 分组取中位数
    key = np.c_[np.round(allpts[:, 3] * FPS),
                np.round(allpts[:, 0] / 10), np.round(allpts[:, 1] / 10)]
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n_uniq = inv.max() + 1
    cnt = np.bincount(inv, minlength=n_uniq)
    dedup = np.zeros((n_uniq, 4))
    for k in range(4):
        dedup[:, k] = np.bincount(inv, weights=allpts[:, k], minlength=n_uniq) / cnt
    if verbose:
        print(f"[去重] {len(allpts)} → {n_uniq} 点")
    return series, dedup, (c0, n)


def measure_direction(series):
    """互谱相位法测传播方向（与 diag_hovmoller_xcorr.py 同款两遍去模糊拟合）。
    返回 (n_u, n_v) 单位向量（指向波的下游），片段对不足时返回 None。"""
    from diag_hovmoller_xcorr import T_WAVE, _phase_pair, fit_cn
    frags = []
    for s in series:
        if len(s) < 30:
            continue
        fr = np.round(s[:, 3] * FPS).astype(int)
        order = np.argsort(fr)
        fr, et = fr[order], s[order, 2]
        _, uniq = np.unique(fr, return_index=True)
        fr, et = fr[uniq], et[uniq]
        if len(fr) >= 64:
            E = np.fft.rfft(et - et.mean())
            fq = np.fft.rfftfreq(len(et), 1 / FPS)
            E[(fq < 0.5) | (fq > 1.2)] = 0.0
            et = np.fft.irfft(E, len(et))
        frags.append((fr, et, (np.median(s[:, 0]), np.median(s[:, 1]))))
    raw = []
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            fr_i, et_i, pi = frags[i]
            fr_j, et_j, pj = frags[j]
            du, dv = pj[0] - pi[0], pj[1] - pi[1]
            sep = np.hypot(du, dv)
            if sep < 300.0:
                continue
            out = _phase_pair(fr_i, et_i, fr_j, et_j)
            if out is None:
                continue
            tau, amp_min, n_win = out
            raw.append((tau, du, dv, amp_min * np.sqrt(n_win), sep))
    if len(raw) < 6:
        return None
    near = [p for p in raw if p[4] < 1000.0]
    c1, n1, _ = fit_cn(near if len(near) >= 3 else raw)
    unw = []
    for tau, du, dv, w, sep in raw:
        pred = (du * n1[0] + dv * n1[1]) / c1
        cands = [tau + k * T_WAVE for k in range(-3, 4)]
        unw.append((min(cands, key=lambda x: abs(x - pred)), du, dv, w))
    c2, n2, _ = fit_cn(unw)
    print(f"[方向] 互谱相位法: c = {c2:.0f} mm/s, 方向 ({n2[0]:.3f}, {n2[1]:.3f}), "
          f"{len(raw)} 对")
    return n2


def main():
    os.makedirs(OUT, exist_ok=True)
    # 可用 argv[1] 指定轨迹 pkl（默认 canonical v2）
    pkl = sys.argv[1] if len(sys.argv) > 1 else PKL
    print(f"[输入] {pkl}")
    series, allpts, (c0, n) = prepare_data(pkl)
    # 传播方向先验：把 (u,v) 旋转到传播坐标系（x'=ξ 沿传播，y'=ζ 垂直），
    # 波场退化为准一维行波 η(ξ,t)，各向异性 Fourier 特征更好分配。
    n_dir = measure_direction(series)
    rotated = n_dir is not None
    R = np.eye(2)
    if rotated:
        R = np.array([[n_dir[0], n_dir[1]], [-n_dir[1], n_dir[0]]])
        allpts = allpts.copy()
        allpts[:, :2] = allpts[:, :2] @ R.T
    eta = allpts[:, 2]

    # 留出 20% 点做泛化测试
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(allpts))
    n_test = len(allpts) // 5
    te, tr = idx[:n_test], idx[n_test:]

    bounds = {"x": (allpts[:, 0].min(), allpts[:, 0].max()),
              "y": (allpts[:, 1].min(), allpts[:, 1].max()),
              "t": (allpts[:, 3].min(), allpts[:, 3].max())}
    xyt = torch.tensor(allpts[tr][:, [0, 1, 3]], dtype=torch.float32)
    eta_t = torch.tensor(allpts[tr][:, 2:3], dtype=torch.float32)
    xyt_te = torch.tensor(allpts[te][:, [0, 1, 3]], dtype=torch.float32)
    eta_te = torch.tensor(allpts[te][:, 2:3], dtype=torch.float32)

    # c 固定为理论值：数据密度下 c 反演不可靠（曾被噪声宽带模态拖到
    # 772 mm/s），而互谱相位法已从数据独立测得 c=1975 mm/s（95% CI
    # [1867,2107]），与深水理论 1976 一致（见 diag_hovmoller_xcorr.py）。
    # 物理损失的作用是约束时空相干性，不再承担测速任务。
    # 传播坐标系下各向异性 sigma：ξ 向含波数 k（归一化后 B~1），
    # ζ 向变化缓慢（0.3）；t 不变。未旋转时退回各向同性默认。
    sigmas = (1.0, 0.3, 1.0) if rotated else (0.5, 0.5, 1.0)
    model = PINNWaveV2(bounds, c_init=C_THEORY, learn_c=False, sigmas=sigmas)
    model = train_pinn(model, xyt, eta_t, bounds, epochs=2500,
                       lambda_phys=1.0, n_colloc=2048, log_every=500,
                       device=DEVICE)

    # 评估：R² = 1 − MSE/Var（相对"预测均值"基线的技能分）
    model.eval()
    with torch.no_grad():
        pred_te = torch.cat([model.predict(xyt_te[i:i + 4096].to(DEVICE))
                             for i in range(0, len(xyt_te), 4096)]).cpu()
    mse = torch.mean((pred_te - eta_te) ** 2).item()
    r2 = 1.0 - mse / eta_te.var().item()
    c_rec = model.c.item()
    print(f"\n[结果] 留出集 R² = {r2:.3f}（>0 才比均值基线强，>0.5 较好）")
    print(f"[结果] c = {c_rec:.0f} mm/s（固定为理论值，反演不可靠见上注释）")

    # 可视化：t=中值帧的预测波面 + 数据散点
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_mid = 0.5 * sum(bounds["t"])
    gx, gy = np.meshgrid(np.linspace(*bounds["x"], 60),
                         np.linspace(*bounds["y"], 60), indexing="ij")
    grid = torch.tensor(np.c_[gx.ravel(), gy.ravel(),
                              np.full(gx.size, t_mid)], dtype=torch.float32)
    with torch.no_grad():
        zg = torch.cat([model.predict(grid[i:i + 8192].to(DEVICE))
                        for i in range(0, len(grid), 8192)]).cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    sc0 = ax[0].scatter(allpts[:, 0], allpts[:, 1], c=eta, cmap="RdBu_r",
                        s=8, vmin=-3 * eta.std(), vmax=3 * eta.std())
    ax[0].set_title("input data (all frames, eta mm)")
    fig.colorbar(sc0, ax=ax[0])
    im = ax[1].pcolormesh(gx, gy, zg.reshape(gx.shape), cmap="RdBu_r",
                          shading="auto", vmin=-3 * eta.std(), vmax=3 * eta.std())
    ax[1].set_title(f"PINN prediction at t = {t_mid:.1f} s (eta mm)")
    fig.colorbar(im, ax=ax[1])
    for a_ in ax[:2]:
        a_.set_xlabel("u (mm)"); a_.set_ylabel("v (mm)"); a_.set_aspect("equal")

    # Hovmöller (u, t)：取 v 中位数附近 ±300mm 带，u 方向 100mm 分箱、
    # t 方向 0.1s 分箱求 η 均值。行波传播呈现对角条纹，
    # 条纹斜率 du/dt 即相速度，可与 c 反演值、理论值三方互证。
    v_med = np.median(allpts[:, 1])
    band = np.abs(allpts[:, 1] - v_med) < 300
    pb = allpts[band]
    if len(pb) > 50:
        ub = np.arange(bounds["x"][0], bounds["x"][1] + 100, 100)
        tb = np.arange(bounds["t"][0], bounds["t"][1] + 0.1, 0.1)
        iu = np.clip(np.digitize(pb[:, 0], ub) - 1, 0, len(ub) - 2)
        it = np.clip(np.digitize(pb[:, 3], tb) - 1, 0, len(tb) - 2)
        H = np.full((len(ub) - 1, len(tb) - 1), np.nan)
        Cnt = np.zeros_like(H, dtype=int)
        Sum = np.zeros_like(H)
        np.add.at(Sum, (iu, it), pb[:, 2])
        np.add.at(Cnt, (iu, it), 1)
        H[Cnt > 0] = Sum[Cnt > 0] / Cnt[Cnt > 0]
        im2 = ax[2].pcolormesh(tb, ub / 1000.0, H, cmap="RdBu_r", shading="auto",
                               vmin=-2 * eta.std(), vmax=2 * eta.std())
        ax[2].set_title(f"Hovmoller (u,t), |v-{v_med:.0f}|<300mm (eta mm)")
        ax[2].set_xlabel("t (s)"); ax[2].set_ylabel("u (m)")
        # 注意：第二个位置参数是 cax（把色条画进指定 axes），
        # 误写 colorbar(im2, ax[2]) 会让色条盖住整个面板——必须用关键字 ax=ax[2]
        fig.colorbar(im2, ax=ax[2])
    else:
        ax[2].text(0.5, 0.5, "band too sparse", ha="center", va="center")
    fig.tight_layout()
    png = os.path.join(OUT, "field_comparison.png")
    fig.savefig(png, dpi=150)
    torch.save({"model": model.state_dict(), "bounds": bounds,
                "plane_centroid": c0, "plane_normal": n,
                # 传播坐标系旋转矩阵（未旋转时为单位阵）：预测前需先把
                # 原始 (u,v) 右乘 rot.T 变换到 ξ-ζ 坐标
                "rot": R if rotated else np.eye(2)},
               os.path.join(OUT, "pinn_real.pt"))
    print(f"[输出] {png} 与 {OUT}/pinn_real.pt 已保存")


if __name__ == "__main__":
    main()
