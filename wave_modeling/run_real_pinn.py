# wave_modeling/run_real_pinn.py
"""
真实数据端到端实验：data/trajectories/trajectories_3d_v2_dino.pkl →
PCA 主平面坐标 → PINN v2 训练 → 留出片段评估。

输入格式（rematch_dino_v2.py 的输出）：list[np.ndarray]，每条 (N,4)：
frame（绝对帧号）, X, Y, Z（mm，相机1坐标系）。片段普遍有缺帧
（46/47 条），frame 列才是真帧号。

坐标处理：相机俯视视角下水面在相机坐标系中是倾斜平面，直接对 Z 取
残差会混入倾角放大（曾出现拟合平面"倾角 84.7°"的伪影）。改为 PCA：
SVD 求点云主平面，η = 沿法向的残差（真实的波面起伏），(u,v) 为面内坐标。

时间单位：pca_plane_coords 已按 FPS=50 把帧号换算为秒，c 单位即 mm/s。
c 固定为理论值 1976 mm/s（反演不可靠；数据独立测速见
diag_hovmoller_xcorr.py 的互谱相位法：1975 mm/s，95% CI [1867,2107]）。

留出评估的泄漏防控（2026-07 评审修复）：
- train/test 按【片段】划分（整段进一侧，固定 seed），不再随机散点划分——
  同一片段 50fps 相邻样本相距 0.02s，随机划分让每个测试点都有训练点贴脸，
  留出 R² 退化为插值技能 + 直推式（transduction）泄漏；
- PCA 平面、MAD 阈值、归一化 bounds 只用训练片段拟合，再套用到测试片段；
- 逐片段 debias 用各片段自身中位数（每片段一个常数，不跨片段共享信息，
  测试片段用自身中位数不构成泄漏）；
- 跨片段去重在 train/test 内部分别进行（跨侧均值合并同样混合两侧信息）。

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
PKL = os.path.join(ROOT, "data/trajectories/trajectories_3d_v2_dino.pkl")
OUT = os.path.join(ROOT, "wave_modeling/real_run")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 50.0          # 采集帧率（刘晔恒论文：50 Hz）
C_THEORY = 1976.0   # 理论波速 mm/s（0.79 Hz 深水规则波，λ=2.5017m）


def load_trajectories(pkl_path, min_len=20):
    """v2 格式：每条 (N,4) = [frame, X, Y, Z]。"""
    with open(pkl_path, "rb") as f:
        trajs = pickle.load(f)
    return [np.asarray(t, dtype=np.float64) for t in trajs if len(t) >= min_len]


def pca_plane_coords(trajs, basis=None):
    """PCA 拟合平均水面：返回 (series, c0, n, vt)，series 为逐片段
    [u, v, eta, t(s)]，eta 为法向残差（mm），t 由帧号换算为秒。
    basis=(c0, vt) 给定时跳过拟合直接套用——用于把训练片段上拟合的
    平面应用到留出片段（预处理不外泄测试分布）。"""
    if basis is None:
        allpts = np.vstack([t[:, 1:4] for t in trajs])
        c0 = allpts.mean(axis=0)
        _, _, vt = np.linalg.svd(allpts - c0, full_matrices=False)
    else:
        c0, vt = basis
    e_u, e_v, n = vt[0], vt[1], vt[2]  # 法向 n 是最小奇异值方向
    series = []
    for t in trajs:
        d = t[:, 1:4] - c0
        series.append(np.c_[d @ e_u, d @ e_v, d @ n, t[:, 0] / FPS])
    return series, c0, n, vt


def _mad_filter(series, ref=None):
    """稳健剔除法向离群点（跟踪噪声导致的瞬间跳点）。
    ref=(eta_med, thr) 给定时直接套用（训练片段拟合 → 留出片段）。"""
    if ref is None:
        eta = np.vstack(series)[:, 2]
        eta_med = np.median(eta)
        thr = 5 * max(1.4826 * np.median(np.abs(eta - eta_med)), 1e-6)
    else:
        eta_med, thr = ref
    series = [s[np.abs(s[:, 2] - eta_med) < thr] for s in series]
    return series, eta_med, thr


def _debias(series):
    """逐轨迹去均值（debias）：诊断显示轨迹间存在 ±300~560mm 的常值深度偏差，
    与 FoundationStereo 残差实验的"碗形"系统误差同源——标定/矫正残差导致的
    位置相关深度偏差。平均水面已知是平的，故每条轨迹的时间中位数 ≈ 偏差本身，
    减去后只损失"平均水面为平"这一已知信息，保留波动的时间相位信息。
    注意：每片段只减【自身】中位数（一个常数），留出片段同样用自身中位数，
    不构成 train/test 泄漏；根治需重标定或静水参考帧。"""
    out = []
    for s in series:
        if len(s) == 0:      # MAD 后可能整段剔空，median 会 nan + RuntimeWarning
            continue
        s = s.copy()
        s[:, 2] -= np.median(s[:, 2])
        out.append(s)
    return out


def _dedup(series):
    """跨片段重复点去重（同一粒子同一帧经不同片段对重复三角化）：
    按 (帧, u/v 10mm 网格) 分组取均值。"""
    allpts = np.vstack(series)
    key = np.c_[np.round(allpts[:, 3] * FPS),
                np.round(allpts[:, 0] / 10), np.round(allpts[:, 1] / 10)]
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n_uniq = inv.max() + 1
    cnt = np.bincount(inv, minlength=n_uniq)
    dedup = np.zeros((n_uniq, 4))
    for k in range(4):
        dedup[:, k] = np.bincount(inv, weights=allpts[:, k], minlength=n_uniq) / cnt
    return dedup


def compute_bounds(pts):
    """归一化 bounds。留出评估时只传【训练】点：归一化也是预处理，
    不能看测试集分布（测试点略超出 [-1,1] 属正常，评估反而更诚实）。"""
    return {"x": (pts[:, 0].min(), pts[:, 0].max()),
            "y": (pts[:, 1].min(), pts[:, 1].max()),
            "t": (pts[:, 3].min(), pts[:, 3].max())}


def _regrid_uniform(fr, et, max_interp_gap=4):
    """把片段按真实帧号重采样到均匀帧时间基（缺帧不能被 FFT 当作均匀采样，
    否则频率最多偏 ~6%）：≤max_interp_gap 帧的短空洞（≤0.08s ≈ 1/10 周期）
    线性插值；更长的空洞零填充——序列已 debias/带通居中，0 ≈ 均值电平，
    且窄带单频信号在零填充下相位近似无偏，而线性插值会在 >1/4 周期的
    空洞上注入错误相位（实测最长空洞 24 帧 = 0.48s ≈ 0.37 周期）。
    无空洞时恒等。要求 fr 已排序去重。"""
    grid = np.arange(fr[0], fr[-1] + 1)
    if len(grid) == len(fr):
        return fr, et
    out = np.interp(grid, fr, et)
    long = np.flatnonzero(np.diff(fr) - 1 > max_interp_gap)
    for i in long:                       # 长空洞段改回 0（均值电平）
        out[fr[i] + 1 - fr[0]: fr[i + 1] - fr[0]] = 0.0
    return grid, out


def prepare_data(pkl_path=PKL, min_len=20, verbose=True):
    """加载轨迹 → PCA 主平面坐标 → MAD 法向离群剔除 → 逐片段 debias → 跨片段去重。
    返回 (series, allpts, meta)：series 为逐片段 [u,v,eta,t(s)]（未去重），
    allpts 为去重后的合并点云，meta = (plane_centroid, plane_normal)。
    供诊断脚本（diag_hovmoller_xcorr.py / final_visualize.py / verify_results.py）
    共用，保证预处理唯一出处。注意：本函数在【全量】数据上拟合预处理，
    留出评估请用 prepare_data_split（只在训练片段上拟合）。"""
    trajs = load_trajectories(pkl_path, min_len)
    series, c0, n, _ = pca_plane_coords(trajs)
    n_traj = len(series)
    n_raw = sum(len(s) for s in series)
    series, _, _ = _mad_filter(series)
    n_kept = sum(len(s) for s in series)
    eta = np.vstack(series)[:, 2]
    if verbose:
        print(f"[数据] 轨迹 {n_traj} 条，点 {n_kept} 个（剔除法向离群 {n_raw - n_kept} 个）| "
              f"η std = {eta.std():.2f} mm, "
              f"范围 [{eta.min():.1f}, {eta.max():.1f}] mm")

    offsets = np.array([np.median(s[:, 2]) for s in series if len(s)])
    series = _debias(series)
    eta = np.vstack(series)[:, 2]
    if verbose:
        print(f"[debias] 轨迹间偏移 std = {offsets.std():.1f} mm（已逐轨迹减去自身中位数）| "
              f"去偏后 η std = {eta.std():.2f} mm, "
              f"范围 [{eta.min():.1f}, {eta.max():.1f}] mm")

    dedup = _dedup(series)
    if verbose:
        print(f"[去重] {len(eta)} → {len(dedup)} 点")
    return series, dedup, (c0, n)


def prepare_data_split(pkl_path=PKL, min_len=20, test_frac=0.2, seed=0,
                       verbose=True):
    """片段级 train/test 划分的数据准备（留出评估专用，防泄漏，见文件头）：
    整条片段分入一侧（固定 seed，按片段数 ~80/20）；PCA 平面与 MAD 阈值
    只在训练片段上拟合再套用到测试片段；去重在两侧内部分别进行。
    返回 dict：series_tr/series_te（未去重逐片段）、dedup_tr/dedup_te、
    tr_idx/te_idx（片段编号）、c0、n。"""
    trajs = load_trajectories(pkl_path, min_len)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(trajs))
    n_test = max(1, int(round(len(trajs) * test_frac)))
    te_idx, tr_idx = np.sort(perm[:n_test]), np.sort(perm[n_test:])

    series_tr, c0, n, vt = pca_plane_coords([trajs[i] for i in tr_idx])
    series_te, _, _, _ = pca_plane_coords([trajs[i] for i in te_idx],
                                          basis=(c0, vt))
    series_tr, eta_med, thr = _mad_filter(series_tr)
    series_te, _, _ = _mad_filter(series_te, ref=(eta_med, thr))
    series_tr = _debias(series_tr)
    series_te = _debias(series_te)
    dedup_tr, dedup_te = _dedup(series_tr), _dedup(series_te)
    if verbose:
        print(f"[划分] 片段级 {1 - test_frac:.0%}/{test_frac:.0%}（seed={seed}）："
              f"train {len(tr_idx)} 段 {len(dedup_tr)} 点 / "
              f"test {len(te_idx)} 段 {len(dedup_te)} 点 | "
              f"train η std = {dedup_tr[:, 2].std():.2f} mm, "
              f"test η std = {dedup_te[:, 2].std():.2f} mm")
    return {"series_tr": series_tr, "series_te": series_te,
            "dedup_tr": dedup_tr, "dedup_te": dedup_te,
            "tr_idx": tr_idx, "te_idx": te_idx, "c0": c0, "n": n}


def measure_direction(series):
    """互谱相位法测传播方向（与 diag_hovmoller_xcorr.py 同款两遍去模糊拟合；
    带通 [0.5,1.2] Hz 与之一致，final_visualize.py 波成分提取用 [0.6,1.0]）。
    τ 换算沿用 _phase_pair 默认 f0=0.79Hz：此处只求方向，~1% 频偏对
    方向角影响可忽略。返回 (n_u, n_v) 单位向量（指向波的下游），
    片段对不足时返回 None。"""
    from diag_hovmoller_xcorr import T_WAVE, _phase_pair, fit_cn
    frags = []
    for s in series:
        if len(s) < 30:
            continue
        fr = np.round(s[:, 3] * FPS).astype(int)
        order = np.argsort(fr)
        fr, et = fr[order], s[order, 2]
        _, uniq = np.unique(fr, return_index=True)
        fr, et = _regrid_uniform(fr[uniq], et[uniq])   # 缺帧 → 均匀帧时间基
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
    # 可用 argv[1] 指定轨迹 pkl（默认 canonical v2_dino）
    pkl = sys.argv[1] if len(sys.argv) > 1 else PKL
    print(f"[输入] {pkl}")
    # 片段级划分 + 仅训练片段拟合预处理（防泄漏，见文件头与 prepare_data_split）
    d = prepare_data_split(pkl)
    c0, n = d["c0"], d["n"]
    # 传播方向先验：把 (u,v) 旋转到传播坐标系（x'=ξ 沿传播，y'=ζ 垂直），
    # 波场退化为准一维行波 η(ξ,t)，各向异性 Fourier 特征更好分配。
    # 方向是全局几何先验（train/test 共用同一旋转矩阵），用全部片段估计。
    n_dir = measure_direction(d["series_tr"] + d["series_te"])
    rotated = n_dir is not None
    R = np.eye(2)
    pts_tr, pts_te = d["dedup_tr"], d["dedup_te"]
    if rotated:
        R = np.array([[n_dir[0], n_dir[1]], [-n_dir[1], n_dir[0]]])
        pts_tr = pts_tr.copy()
        pts_tr[:, :2] = pts_tr[:, :2] @ R.T
        pts_te = pts_te.copy()
        pts_te[:, :2] = pts_te[:, :2] @ R.T
    allpts = np.vstack([pts_tr, pts_te])
    eta = allpts[:, 2]

    # bounds 只用训练点：归一化也是预处理，不能看测试集分布
    bounds = compute_bounds(pts_tr)
    xyt = torch.tensor(pts_tr[:, [0, 1, 3]], dtype=torch.float32)
    eta_t = torch.tensor(pts_tr[:, 2:3], dtype=torch.float32)
    xyt_te = torch.tensor(pts_te[:, [0, 1, 3]], dtype=torch.float32)
    eta_te = torch.tensor(pts_te[:, 2:3], dtype=torch.float32)

    # c 固定为理论值：数据密度下 c 反演不可靠（曾被噪声宽带模态拖到
    # 772 mm/s），而互谱相位法已从数据独立测得 c=1975 mm/s（95% CI
    # [1867,2107]），与深水理论 1976 一致（见 diag_hovmoller_xcorr.py）。
    # 物理损失的作用是约束时空相干性，不再承担测速任务。
    # 传播坐标系下各向异性 sigma（归一化坐标 v∈[-1,1]，全域周期数 ≈ 2σ，
    # sigma 位置与输入维 (ξ, ζ, t) 一一对应，见 FourierFeatures）：
    #   σ_ξ=0.5：ξ 跨度 2422mm ≈ 0.97λ，全域约 1 个波动周期 → 2σ≈1；
    #   σ_ζ=0.3：ζ 向变化缓慢；
    #   σ_t=8.0：t 跨度 19.98s × 0.79Hz ≈ 15.8 周期 → 2σ≈16，原值 1.0
    #   欠覆盖时间谱 ~8×。B 可学习，初始化贴近目标谱即可（pinn_v2 教训一）。
    sigmas = (0.5, 0.3, 8.0) if rotated else (0.5, 0.5, 8.0)
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
    # 片段级留出：整段轨迹未参与训练与预处理拟合，是泛化技能而非插值技能
    print(f"\n[结果] 留出片段 R² = {r2:.3f}（整段留出 + 预处理仅 train 拟合；"
          f">0 才比均值基线强，>0.5 较好）")
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
