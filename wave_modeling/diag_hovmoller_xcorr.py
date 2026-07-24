# wave_modeling/diag_hovmoller_xcorr.py
"""
Hovmöller 复核 + 轨迹互相关独立测量相速度（不依赖 PINN）。

背景：run_real_pinn.py 的 Hovmöller 面板（u 100mm × t 0.1s 分箱）对稀疏
轨迹数据而言大部分格应为空，但图上看似"画得太满"，且只见沿 u 的静态梯度、
无对角传播条纹。本脚本做三件事：

A. 打印分箱占用率统计，用粗分箱（250mm × 0.5s）重画数据 Hovmöller，
   并在同一 (u,t)、(v,t) 网格上叠加 PINN 预测场对比（预测场是否有条纹）。

B. 互谱相位法测 c：每条轨迹（泡沫粒子）水平位置近似不动，其 η(t) 是
   0.79 Hz 单频振荡，相位由沿传播方向的位置决定。对时间重叠充分的
   轨迹对，在重叠窗口内求 0.79 Hz 处互谱相位差 → 时滞
   τ_ij = Δx_ij·n/c，多对加权最小二乘解出相速度大小 c 与传播方向 n。
   与理论值 1976 mm/s 独立互证。
   周期模糊（τ±kT）用两遍展开：先近距对（|τ|<T/2 无模糊）拟合初值，
   再对全部对选离预测最近的候选时滞重新拟合。
   （曾用互相关 argmax 估计 τ：带通后信号接近纯正弦，余弦平台期导致
   argmax 在不同周期间跳动，c 从 2515 漂到 3706 mm/s，故弃用。）

用法：.venv_fs/Scripts/python.exe wave_modeling/diag_hovmoller_xcorr.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_real_pinn import FPS, C_THEORY, OUT, prepare_data  # noqa: E402

F_WAVE = 0.79          # 规则波频率 Hz（论文）
T_WAVE = 1.0 / F_WAVE  # 周期 s
MIN_OVERLAP_S = 1.5    # 相位估计最少重叠时长（>1 个周期）
MIN_SEP = 300.0        # 轨迹对最小空间间距 mm
NEAR_SEP = 1000.0      # 第一遍无模糊拟合的最大间距 mm


# ---------------------------------------------------------------- A. Hovmöller
def _bin2d(pts, u_edges, t_edges):
    """pts: [N,3] = (coord, eta, t)。返回 (H, Cnt)。"""
    iu = np.clip(np.digitize(pts[:, 0], u_edges) - 1, 0, len(u_edges) - 2)
    it = np.clip(np.digitize(pts[:, 2], t_edges) - 1, 0, len(t_edges) - 2)
    H = np.full((len(u_edges) - 1, len(t_edges) - 1), np.nan)
    Cnt = np.zeros_like(H, dtype=int)
    Sum = np.zeros_like(H)
    np.add.at(Sum, (iu, it), pts[:, 1])
    np.add.at(Cnt, (iu, it), 1)
    H[Cnt > 0] = Sum[Cnt > 0] / Cnt[Cnt > 0]
    return H, Cnt


def diag_hovmoller(series, allpts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bounds = {"x": (allpts[:, 0].min(), allpts[:, 0].max()),
              "y": (allpts[:, 1].min(), allpts[:, 1].max()),
              "t": (allpts[:, 3].min(), allpts[:, 3].max())}
    eta_std = allpts[:, 2].std()
    v_med, u_med = np.median(allpts[:, 1]), np.median(allpts[:, 0])

    # ---- 细分箱占用率统计（解答"画得太满"的疑问）----
    band_u = np.abs(allpts[:, 1] - v_med) < 300
    pu = np.c_[allpts[band_u, 0], allpts[band_u, 2], allpts[band_u, 3]]
    ub_f = np.arange(bounds["x"][0], bounds["x"][1] + 100, 100)
    tb_f = np.arange(bounds["t"][0], bounds["t"][1] + 0.1, 0.1)
    Hf, Cf = _bin2d(pu, ub_f, tb_f)
    occ = np.mean(Cf.ravel() > 0)
    print(f"[Hovmöller] u 带内点数 {len(pu)}/{len(allpts)} | "
          f"细分箱 {Cf.size} 格，占用率 {occ:.1%}，"
          f"非空格点数中位数 {np.median(Cf[Cf > 0]):.0f}")

    # ---- 粗分箱 (u,t) 与 (v,t) ----
    ub_c = np.arange(bounds["x"][0], bounds["x"][1] + 250, 250)
    vb_c = np.arange(bounds["y"][0], bounds["y"][1] + 250, 250)
    tb_c = np.arange(bounds["t"][0], bounds["t"][1] + 0.5, 0.5)
    Hu, Cu = _bin2d(pu, ub_c, tb_c)
    band_v = np.abs(allpts[:, 0] - u_med) < 300
    pv = np.c_[allpts[band_v, 1], allpts[band_v, 2], allpts[band_v, 3]]
    Hv, Cv = _bin2d(pv, vb_c, tb_c)
    print(f"[Hovmöller] 粗分箱 (u,t) 占用率 {np.mean(Cu.ravel() > 0):.1%} | "
          f"(v,t) 占用率 {np.mean(Cv.ravel() > 0):.1%}")

    # ---- PINN 预测场（同一网格，v=v_med / u=u_med 截面）----
    Pu = Pv = None
    ckpt_path = os.path.join(OUT, "pinn_real.pt")
    if os.path.exists(ckpt_path):
        import torch
        from pinn_v2 import PINNWaveV2
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # 模型在传播坐标系 (ξ,ζ) 上训练：物理 (u,v) 查询点须先右乘 rot.T
        # （与 final_visualize.py / verify_results.py 同一约定）；
        # 旧 checkpoint 无 rot 键（方向先验引入前训练）时退化为单位阵。
        rot = ck.get("rot", np.eye(2))
        model = PINNWaveV2(ck["bounds"], c_init=C_THEORY)
        model.load_state_dict(ck["model"])
        model.eval()
        uc = 0.5 * (ub_c[:-1] + ub_c[1:])
        vc = 0.5 * (vb_c[:-1] + vb_c[1:])
        tc = 0.5 * (tb_c[:-1] + tb_c[1:])
        with torch.no_grad():
            qu = np.c_[np.repeat(uc, len(tc)),
                       np.full(uc.size * len(tc), v_med),
                       np.tile(tc, len(uc))]
            qu[:, :2] = qu[:, :2] @ rot.T   # (u,v) → (ξ,ζ)
            Pu = model.predict(torch.tensor(qu, dtype=torch.float32)
                               ).numpy().reshape(len(uc), len(tc))
            qv = np.c_[np.full(vc.size * len(tc), u_med),
                       np.repeat(vc, len(tc)),
                       np.tile(tc, len(vc))]
            qv[:, :2] = qv[:, :2] @ rot.T   # (u,v) → (ξ,ζ)
            Pv = model.predict(torch.tensor(qv, dtype=torch.float32)
                               ).numpy().reshape(len(vc), len(tc))
        # 数据面板与预测面板在有效格上的相关（检验 PINN 是否复现数据结构）
        m = ~np.isnan(Hu)
        if m.sum() > 10:
            r = np.corrcoef(Hu[m], Pu[m])[0, 1]
            print(f"[Hovmöller] 数据 vs PINN 面板相关（u,t 粗分箱有效格）: r = {r:.3f}")

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(2, 3, figsize=(19, 8))
    vlim = 2 * eta_std

    im = ax[0, 0].pcolormesh(tb_f, ub_f / 1000, Hf, cmap=cmap, shading="auto",
                             vmin=-vlim, vmax=vlim)
    ax[0, 0].set_title(f"data (u,t) fine bins 100mm/0.1s, occ {occ:.1%}")
    im = ax[0, 1].pcolormesh(tb_c, ub_c / 1000, Hu, cmap=cmap, shading="auto",
                             vmin=-vlim, vmax=vlim)
    ax[0, 1].set_title("data (u,t) coarse 250mm/0.5s")
    im = ax[0, 2].pcolormesh(tb_c, ub_c / 1000, Cu, cmap="Greens", shading="auto")
    ax[0, 2].set_title("occupancy (points per bin)")
    fig.colorbar(im, ax=ax[0, 2])
    if Pu is not None:
        im = ax[1, 0].pcolormesh(tb_c, ub_c / 1000, Pu, cmap=cmap, shading="auto",
                                 vmin=-vlim, vmax=vlim)
        ax[1, 0].set_title(f"PINN (u,t) at v={v_med:.0f}mm")
        im = ax[1, 1].pcolormesh(tb_c, vb_c / 1000, Hv, cmap=cmap, shading="auto",
                                 vmin=-vlim, vmax=vlim)
        ax[1, 1].set_title("data (v,t) coarse")
        im = ax[1, 2].pcolormesh(tb_c, vb_c / 1000, Pv, cmap=cmap, shading="auto",
                                 vmin=-vlim, vmax=vlim)
        ax[1, 2].set_title(f"PINN (v,t) at u={u_med:.0f}mm")
    for a in ax.ravel():
        a.set_xlabel("t (s)")
    for a in ax[:, 0]:
        a.set_ylabel("coord (m)")
    fig.tight_layout()
    png = os.path.join(OUT, "diag_hovmoller.png")
    fig.savefig(png, dpi=140)
    print(f"[输出] {png}")


# ---------------------------------------------------------------- B. 互相关测 c
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


def _frag_series(series):
    """每条片段 → (帧号数组, η 数组, (u_med, v_med))，帧号整数。
    ≥64 帧的片段做 [0.5,1.2] Hz 带通（压制残余偏差趋势与高频噪声，
    突出 0.79 Hz 波峰的相位信息）；更短的片段频率分辨率不足，保持原样。
    注意：同款带通同源复制于 run_real_pinn.measure_direction（[0.5,1.2]），
    final_visualize.py 波成分提取用 [0.6,1.0]——改动任一处请对照其余两处。"""
    out = []
    for s in series:
        if len(s) < 30:
            continue
        fr = np.round(s[:, 3] * FPS).astype(int)
        order = np.argsort(fr)
        fr, et = fr[order], s[order, 2]
        _, uniq = np.unique(fr, return_index=True)
        fr, et = fr[uniq], et[uniq]
        fr, et = _regrid_uniform(fr, et)   # 缺帧 → 均匀帧时间基
        if len(fr) >= 64:
            E = np.fft.rfft(et - et.mean())
            fq = np.fft.rfftfreq(len(et), 1 / FPS)
            E[(fq < 0.5) | (fq > 1.2)] = 0.0
            et = np.fft.irfft(E, len(et))
        out.append((fr, et, (np.median(s[:, 0]), np.median(s[:, 1]))))
    return out


def _phase_pair(fr_i, et_i, fr_j, et_j, f0=F_WAVE):
    """重叠窗口内 f0 处互谱相位差 → 时滞 τ = Δφ/(2π·f0)（主值，mod 1/f0）。
    f0 默认论文值 0.79 Hz；调用方应传实测主峰中位（零填充精细测频，
    本数据 ≈0.781 Hz）——写死 0.79 会引入 ~1% 的 c 系统偏差。
    两序列取同一重叠窗口、同一 nfft（零填充 4096，df≈0.012 Hz），
    相位差即互谱 arg(A·conj(B))，等效于用全部重叠样本的最优相位估计，
    不受余弦平台期 argmax 跳变影响。
    返回 (tau[s], amp_min[mm], n_window) 或 None。"""
    w0 = max(fr_i[0], fr_j[0])
    w1 = min(fr_i[-1], fr_j[-1])
    n = int(w1 - w0 + 1)
    if n < int(MIN_OVERLAP_S * FPS):
        return None
    a = np.zeros(n)
    b = np.zeros(n)
    ai = fr_i - w0
    bj = fr_j - w0
    va = (ai >= 0) & (ai < n)
    vb = (bj >= 0) & (bj < n)
    a[ai[va]] = et_i[va]
    b[bj[vb]] = et_j[vb]
    a -= a.mean()
    b -= b.mean()
    nfft = 4096
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    k = int(round(f0 * nfft / FPS))
    amp_a = 2 * abs(A[k]) / n
    amp_b = 2 * abs(B[k]) / n
    if amp_a < 4.0 or amp_b < 4.0:   # 两条片段在 f0 都真得有波才行
        return None
    tau = np.angle(A[k] * np.conj(B[k])) / (2 * np.pi * f0)
    return tau, min(amp_a, amp_b), n


def fit_cn(pairs):
    """pairs: list of (tau[s], du[mm], dv[mm], w)。WLS 解 τ = (Δu,Δv)·n/c，
    返回 (c[mm/s], n[2], 时滞残差[s])。"""
    X = np.array([[p[1], p[2]] for p in pairs])
    y = np.array([p[0] for p in pairs])
    sw = np.sqrt(np.array([p[3] for p in pairs]))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    res = y - X @ beta
    c = 1.0 / np.linalg.norm(beta)       # |β| = 1/c [s/mm] → c [mm/s]
    nvec = beta * c
    return c, nvec, res


def diag_xcorr(series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frags = _frag_series(series)
    print(f"[xcorr] 片段 {len(frags)} 条（≥30 帧）")

    # 每条片段 FFT：确认 0.79 Hz 主峰普遍存在（只用 ≥100 帧的片段；
    # 零填充 8192 精细定位峰频——短片段未零填充的 df 达 0.5Hz，
    # 直接 argmax 测到的只是粗 bin 中心而非音调）
    pk_f, pk_a = [], []
    for fr, et, _ in frags:
        if len(fr) < 100:
            continue
        e = et - et.mean()
        w = np.hanning(len(e))
        sp = np.abs(np.fft.rfft(e * w, 8192))
        fq = np.fft.rfftfreq(8192, 1 / FPS)
        k = np.argmax(sp[1:]) + 1
        pk_f.append(fq[k])
        pk_a.append(2 * sp[k] / w.sum())  # 窗增益修正：幅值 = 2|X|/Σw（Hann 即 4|X|/N）
    pk_f, pk_a = np.array(pk_f), np.array(pk_a)
    # τ = Δφ/(2π·f) 用实测主峰中位（本数据 ≈0.781Hz），写死论文值 0.79
    # 会引入 ~1% 的 c 系统偏差；长片段不足时退回论文值 F_WAVE
    f_meas = float(np.median(pk_f)) if len(pk_f) else F_WAVE
    if len(pk_f):
        print(f"[xcorr] ≥100 帧片段 {len(pk_f)} 条 | FFT 主峰频率中位 "
              f"{f_meas:.3f} Hz（IQR {np.percentile(pk_f, 25):.3f}–"
              f"{np.percentile(pk_f, 75):.3f}）| 振幅中位 {np.median(pk_a):.1f} mm")
    else:
        print(f"[xcorr] 无 ≥100 帧片段，τ 换算退回论文值 {F_WAVE} Hz")

    raw = []
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            fr_i, et_i, pi = frags[i]
            fr_j, et_j, pj = frags[j]
            du = pj[0] - pi[0]
            dv = pj[1] - pi[1]
            sep = np.hypot(du, dv)
            if sep < MIN_SEP:
                continue
            out = _phase_pair(fr_i, et_i, fr_j, et_j, f0=f_meas)
            if out is None:
                continue
            tau, amp_min, n_win = out
            w = amp_min * np.sqrt(n_win)
            raw.append((tau, du, dv, w, sep, amp_min))
    print(f"[phase] 有效轨迹对 {len(raw)} 个（{f_meas:.3f}Hz 幅值≥4mm, "
          f"间距≥{MIN_SEP:.0f}mm）")
    if len(raw) < 6:
        print("[xcorr] 有效对太少，无法拟合 c。")
        return

    # 第一遍：近距对（|τ|<T/2 无周期模糊）拟合初值
    # （两遍去模糊拟合同源复制于 run_real_pinn.measure_direction 与
    #  eval_tracks.evaluate，带通均 [0.5,1.2]Hz；final_visualize 用 [0.6,1.0]）
    near = [p for p in raw if p[4] < NEAR_SEP]
    if near:
        print(f"[phase] 近距对（<{NEAR_SEP:.0f}mm）{len(near)} 个 | "
              f"时滞中位 {np.median([abs(p[0]) for p in near]):.3f} s")
    else:
        print(f"[phase] 近距对（<{NEAR_SEP:.0f}mm）0 个，第一遍用全部对兜底")
    if len(near) >= 3:
        c1, n1, _ = fit_cn(near)
    else:  # 近距对不足时用全部对兜底
        c1, n1, _ = fit_cn(raw)
    print(f"[phase] 第一遍: c = {c1:.0f} mm/s = {c1 / 1000:.3f} m/s, "
          f"方向 ({n1[0]:.2f}, {n1[1]:.2f})")

    # 第二遍：全体对按 τ±kT 展开（选离第一遍预测最近的候选）后重拟合
    unwrapped = []
    for tau, du, dv, w, sep, r in raw:
        pred = (du * n1[0] + dv * n1[1]) / c1   # s
        cands = [tau + k * T_WAVE for k in range(-3, 4)]
        tau_u = min(cands, key=lambda x: abs(x - pred))
        unwrapped.append((tau_u, du, dv, w))
    c2, n2, res = fit_cn(unwrapped)
    theta = np.degrees(np.arctan2(n2[1], n2[0]))
    print(f"[phase] 第二遍（全对展开）: c = {c2:.0f} mm/s = "
          f"{c2 / 1000:.3f} m/s | 方向角 {theta:.1f}°（PCA u-v 面内）| "
          f"时滞残差 RMS {np.sqrt(np.mean(res ** 2)) * 1000:.0f} ms")
    print(f"[phase] 深水理论 {C_THEORY:.0f} mm/s | 偏差 "
          f"{abs(c2 - C_THEORY) / C_THEORY:.1%}")

    # 自助法（重采样轨迹对）估计 c 的不确定度
    rng = np.random.default_rng(1)
    cs = np.array([fit_cn([unwrapped[k] for k in
                           rng.integers(len(unwrapped), size=len(unwrapped))])[0]
                   for _ in range(200)])
    print(f"[phase] bootstrap c 95% CI: [{np.percentile(cs, 2.5):.0f}, "
          f"{np.percentile(cs, 97.5):.0f}] mm/s（中位 {np.median(cs):.0f}）")

    # 图：片段分布 / τ-投影距离散点 / 近距对时滞与主峰频率分布
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    pos = np.array([f[2] for f in frags])
    ax[0].scatter(pos[:, 0] / 1000, pos[:, 1] / 1000, s=30)
    ax[0].quiver(pos[:, 0].mean() / 1000, pos[:, 1].mean() / 1000,
                 n2[0], n2[1], angles="xy", scale_units="xy", scale=0.5,
                 color="r", width=0.008)
    ax[0].set_title(f"fragments & wave dir {theta:.0f} deg")
    ax[0].set_xlabel("u (m)"); ax[0].set_ylabel("v (m)"); ax[0].set_aspect("equal")
    proj = np.array([(p[1] * n2[0] + p[2] * n2[1]) / 1000 for p in unwrapped])
    taus = np.array([p[0] for p in unwrapped])
    ax[1].scatter(proj, taus, s=25, alpha=0.7)
    xs = np.linspace(proj.min(), proj.max(), 10)
    ax[1].plot(xs, xs * 1000 / c2, "r-", label=f"fit c={c2:.0f} mm/s")
    ax[1].plot(xs, xs / (C_THEORY / 1000), "g--", label=f"deep-water theory {C_THEORY:.0f} mm/s")
    ax[1].set_xlabel("projected separation (m)"); ax[1].set_ylabel("lag tau (s)")
    ax[1].legend(); ax[1].set_title("lag vs separation along wave dir")
    if len(near):
        ax[2].hist([p[0] for p in near], bins=20, alpha=0.7, label="near-pair tau")
        ax[2].axvline(np.median([p[0] for p in near]), color="r", ls="-")
        ax[2].set_xlabel("near-pair lag tau (s)")
        ax[2].set_ylabel("pair count")
    ax2 = ax[2].twinx()
    if len(pk_f):
        ax2.hist(pk_f, bins=20, alpha=0.4, color="g", label="frag peak f")
        ax2.axvline(F_WAVE, color="g", ls="--")
        ax2.set_ylabel("frag count")
    ax[2].set_title("near-pair lags & peak-frequency dist")
    fig.tight_layout()
    png = os.path.join(OUT, "diag_xcorr.png")
    fig.savefig(png, dpi=140)
    print(f"[输出] {png}")


def main():
    os.makedirs(OUT, exist_ok=True)
    series, allpts, _ = prepare_data()
    diag_hovmoller(series, allpts)
    diag_xcorr(series)


if __name__ == "__main__":
    main()
