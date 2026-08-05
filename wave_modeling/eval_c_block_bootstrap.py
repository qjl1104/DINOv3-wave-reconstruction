# wave_modeling/eval_c_block_bootstrap.py
"""
互谱波速 c 的诚实置信区间：片段级 cluster bootstrap。

现状问题：eval_tracks.py（[相] 指标）与 diag_hovmoller_xcorr.py 的 bootstrap
以『片段对』为重采样单元（200 次），但每条片段参与数十个对，对间强相关，
导致 CI 过窄（基线 CI[2007,2008] mm/s，显然不可信）。

本脚本以片段（轨迹）为重采样单元做 cluster bootstrap：有放回抽取片段
（多重集），对 (i,j) 以其两端片段的重数之积 m_i*m_j 为权重倍数进入 WLS
拟合（同一对在新样本中出现 m_i*m_j 次 ⇔ 权重乘 m_i*m_j，数学上严格等价）。
重复 400 次得片段级 95% CI；并给出按时间分块（block）的变体（块内片段
共享同一次抽中/不中的命运，捕捉波场时间局域异常造成的跨片段相关），
与原版对级 CI 并排对比。

物理部分零重写：_phase_pair / fit_cn / F_WAVE 从
diag_hovmoller_xcorr import；load_series / _regrid_uniform 从 eval_tracks
import。MAD 剔除 + 逐片段 debias + 带通片段构建 + 实测主峰换算频率 +
两遍去周期模糊，是 eval_tracks.evaluate 的逐行同源复制（其未封装成
可 import 的函数）；对级 bootstrap 亦按 eval_tracks 原样（同种子同次数）
复现。（2026-08-04 口径修正后——解缠绕周期 1/f_meas + 测频抛物线插值——
对级 CI 现为 [1991,1993]，文中 CI[2007,2008] 为修正前历史口径。）

用法：.venv_fs/Scripts/python.exe wave_modeling/eval_c_block_bootstrap.py [pkl]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_hovmoller_xcorr import F_WAVE, _phase_pair, fit_cn, peak_freq_interp  # noqa: E402
from eval_tracks import load_series, _regrid_uniform  # noqa: E402
from run_real_pinn import FPS, C_THEORY, PKL, OUT  # noqa: E402

MIN_SEP = 300.0    # 同 eval_tracks
NEAR_SEP = 1000.0  # 同 eval_tracks

N_BOOT_PAIR = 200    # 与 eval_tracks 一致（对级，复现基线）
N_BOOT_SEG = 400     # 片段级 cluster bootstrap 次数（任务要求 300-500）
N_BOOT_BLK = 400     # 时间块变体次数
SEED_PAIR = 1        # 与 eval_tracks 一致
SEED_SEG = 20260730
SEED_BLK = 20260731


def build_pairs(pkl_path):
    """eval_tracks.evaluate 数据通路同源复制：load_series → MAD 剔除 →
    逐片段 debias → 带通片段 → 实测主峰频率 → 相位对（保留片段索引）。
    返回 (frags, pairs)，pairs 为 dict 的等长数组：I, J, TAU, DU, DV, W, SEP。"""
    series = load_series(pkl_path)

    # ---- 同源复制 eval_tracks.evaluate: MAD 离群剔除 + 逐片段 debias ----
    allpts = np.vstack(series)
    eta = allpts[:, 2]
    mad = 1.4826 * np.median(np.abs(eta - np.median(eta)))
    thr = 5 * max(mad, 1e-6)
    series = [s[np.abs(s[:, 2] - np.median(eta)) < thr] for s in series]
    series = [s for s in series if len(s) > 0]
    if not series:
        raise RuntimeError("MAD 剔除后无剩余片段")
    for s in series:
        s[:, 2] -= np.median(s[:, 2])

    # ---- 同源复制 eval_tracks.evaluate: 带通片段构建（[0.5,1.2]Hz）----
    frags = []
    for s in series:
        if len(s) < 30:
            continue
        fr = np.round(s[:, 3] * FPS).astype(int)
        order = np.argsort(fr)
        fr, et = fr[order], s[order, 2]
        _, uniq = np.unique(fr, return_index=True)
        fr, et = _regrid_uniform(fr[uniq], et[uniq])
        if len(fr) >= 64:
            E = np.fft.rfft(et - et.mean())
            fq = np.fft.rfftfreq(len(et), 1 / FPS)
            E[(fq < 0.5) | (fq > 1.2)] = 0.0
            et = np.fft.irfft(E, len(et))
        frags.append((fr, et, (np.median(s[:, 0]), np.median(s[:, 1]))))

    # ---- 同源复制 eval_tracks.evaluate: 实测主峰中位（带通后、零填充 8192
    # + 抛物线插值去栅格量化偏差）----
    pk_bp = []
    for fr, et, _ in frags:
        if len(fr) < 100:
            continue
        e = et - et.mean()
        sp = np.abs(np.fft.rfft(e * np.hanning(len(e)), 8192))
        fq = np.fft.rfftfreq(8192, 1 / FPS)
        band = np.flatnonzero((fq >= 0.5) & (fq <= 1.2))
        k = band[np.argmax(sp[band])]
        pk_bp.append(peak_freq_interp(sp, fq, k))
    f_meas = float(np.median(pk_bp)) if pk_bp else F_WAVE

    # ---- 同源复制 eval_tracks.evaluate: 相位对枚举（多记片段索引 I,J）----
    I, J, TAU, DU, DV, W, SEP = [], [], [], [], [], [], []
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            fr_i, et_i, pi = frags[i]
            fr_j, et_j, pj = frags[j]
            du, dv = pj[0] - pi[0], pj[1] - pi[1]
            if np.hypot(du, dv) < MIN_SEP:
                continue
            out = _phase_pair(fr_i, et_i, fr_j, et_j, f0=f_meas)
            if out is None:
                continue
            tau, amp_min, n_win = out
            I.append(i)
            J.append(j)
            TAU.append(tau)
            DU.append(du)
            DV.append(dv)
            W.append(amp_min * np.sqrt(n_win))
            SEP.append(np.hypot(du, dv))
    pairs = dict(I=np.array(I), J=np.array(J), TAU=np.array(TAU),
                 DU=np.array(DU), DV=np.array(DV), W=np.array(W),
                 SEP=np.array(SEP))
    return frags, pairs, f_meas


def two_pass_unwrap(pairs, f_meas):
    """同源复制 eval_tracks.evaluate 的两遍去周期模糊：
    近距对（<NEAR_SEP）拟合初值 → 全体对 τ±kT 选最近候选展开
    （候选周期用实测频率 1/f_meas，口径修正见
    real_run/diag_c_bias_budget_results.txt §4）。
    返回 (c2, n2, res_rms_ms, TAU_U)，TAU_U 为展开后的时滞数组。"""
    near = pairs["SEP"] < NEAR_SEP
    if near.sum() >= 3:
        c1, n1, _ = fit_cn(list(zip(pairs["TAU"][near], pairs["DU"][near],
                                    pairs["DV"][near], pairs["W"][near])))
    else:
        c1, n1, _ = fit_cn(list(zip(pairs["TAU"], pairs["DU"],
                                    pairs["DV"], pairs["W"])))
    tau_u = np.empty_like(pairs["TAU"])
    for k, (tau, du, dv) in enumerate(zip(pairs["TAU"], pairs["DU"],
                                          pairs["DV"])):
        pred = (du * n1[0] + dv * n1[1]) / c1
        cands = [tau + kk * (1.0 / f_meas) for kk in range(-3, 4)]
        tau_u[k] = min(cands, key=lambda x: abs(x - pred))
    c2, n2, res = fit_cn(list(zip(tau_u, pairs["DU"], pairs["DV"],
                                  pairs["W"])))
    return c2, n2, float(np.sqrt(np.mean(res ** 2)) * 1000.0), tau_u


def bootstrap_pair(tau_u, pairs, n_rep, seed):
    """对级 bootstrap：与 eval_tracks.evaluate 逐行一致（同种子同次数）。"""
    unwrapped = list(zip(tau_u, pairs["DU"], pairs["DV"], pairs["W"]))
    rng = np.random.default_rng(seed)
    return np.array([fit_cn([unwrapped[k] for k in
                             rng.integers(len(unwrapped),
                                          size=len(unwrapped))])[0]
                     for _ in range(n_rep)])


def _fit_with_multiplicity(tau_u, pairs, seg_cnt):
    """给定每条片段的重数 seg_cnt，对 (i,j) 以 m_i*m_j 倍权重进入 fit_cn。
    数学等价：同一行在 WLS 中重复 m 次 ⇔ 权重乘 m（fit_cn 内 sw=√w）。"""
    mult = seg_cnt[pairs["I"]] * seg_cnt[pairs["J"]]
    m = mult > 0
    if m.sum() < 6:        # 与 eval_tracks 的最少对数门槛一致
        return None
    c, _, _ = fit_cn(list(zip(tau_u[m], pairs["DU"][m], pairs["DV"][m],
                              pairs["W"][m] * mult[m])))
    return c


def bootstrap_segment(tau_u, pairs, n_seg, n_rep, seed):
    """片段级 cluster bootstrap：有放回抽 n_seg 条片段。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        cnt = np.bincount(rng.integers(0, n_seg, size=n_seg),
                          minlength=n_seg)
        c = _fit_with_multiplicity(tau_u, pairs, cnt)
        if c is not None:
            out.append(c)
    return np.array(out)


def bootstrap_block(tau_u, pairs, blk_of_seg, n_rep, seed):
    """时间块 bootstrap：有放回抽 K 个时间块，块内全部片段共享块重数。"""
    n_blk = int(blk_of_seg.max()) + 1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        bcnt = np.bincount(rng.integers(0, n_blk, size=n_blk),
                           minlength=n_blk)
        c = _fit_with_multiplicity(tau_u, pairs, bcnt[blk_of_seg])
        if c is not None:
            out.append(c)
    return np.array(out)


def _ci(cs):
    return (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5)))


def main():
    pkl = sys.argv[1] if len(sys.argv) > 1 else PKL
    print(f"[输入] {pkl}")
    frags, pairs, f_meas = build_pairs(pkl)
    n_seg = len(frags)
    P = len(pairs["I"])
    print(f"[数据] 片段 {n_seg} 条（≥30 帧）| 有效片段对 {P} 个"
          f"（{f_meas:.3f}Hz 幅值≥4mm, 间距≥{MIN_SEP:.0f}mm）")

    # ---- 诊断 1：每片段参与多少个对（对间相关的直接证据）----
    deg = np.bincount(np.r_[pairs["I"], pairs["J"]], minlength=n_seg)
    print(f"[相关] 每片段参与对数：中位 {np.median(deg):.0f} | "
          f"均值 {deg.mean():.0f} | max {deg.max()} | "
          f"参与≥50 个对的片段 {(deg >= 50).sum()}/{n_seg}")

    # ---- 诊断 2：片段的时间聚簇（决定是否值得做时间块变体）----
    f_med = np.array([np.median(fr) for fr, _, _ in frags])
    f_lo, f_hi = int(f_med.min()), int(f_med.max())
    print(f"[时间] 片段中位帧范围 [{f_lo}, {f_hi}]（跨度 "
          f"{(f_hi - f_lo) / FPS:.1f}s ≈ {(f_hi - f_lo) / (FPS / f_meas):.0f} 周期）")
    edges = np.linspace(f_lo, f_hi + 1, 21)
    cnt_t, _ = np.histogram(f_med, bins=edges)
    for k in range(20):
        bar = "#" * int(60 * cnt_t[k] / max(cnt_t.max(), 1))
        print(f"       {edges[k]:7.0f}-{edges[k + 1]:7.0f} | {cnt_t[k]:4d} {bar}")

    # ---- 点估计（应复现基线 c≈2008）----
    c2, n2, res_ms, tau_u = two_pass_unwrap(pairs, f_meas)
    print(f"[拟合] 两遍去模糊 c = {c2:.0f} mm/s（深水理论 {C_THEORY:.0f}，"
          f"偏差 {abs(c2 - C_THEORY) / C_THEORY:.1%}）| 方向 "
          f"({n2[0]:.2f},{n2[1]:.2f}) | 时滞残差 RMS {res_ms:.0f} ms")

    # ---- 三种 bootstrap ----
    cs_pair = bootstrap_pair(tau_u, pairs, N_BOOT_PAIR, SEED_PAIR)
    cs_seg = bootstrap_segment(tau_u, pairs, n_seg, N_BOOT_SEG, SEED_SEG)

    # 时间块：按片段中位帧分块，块长 = round(2 个波动周期)
    period_fr = FPS / f_meas
    rows = []
    cs_blk_main = None
    for mult_T in (1, 2, 4):
        blk_len = mult_T * period_fr
        blk = np.floor((f_med - f_lo) / blk_len).astype(int)
        cs_b = bootstrap_block(tau_u, pairs, blk, N_BOOT_BLK, SEED_BLK)
        lo, hi = _ci(cs_b)
        rows.append((f"时间块 bootstrap（块长 {mult_T}T≈"
                     f"{blk_len:.0f}帧, K={int(blk.max()) + 1}）",
                     len(cs_b), float(np.median(cs_b)), lo, hi))
        if mult_T == 2:
            cs_blk_main = cs_b

    lo_p, hi_p = _ci(cs_pair)
    lo_s, hi_s = _ci(cs_seg)

    # ---- 对比表 ----
    print("\n========== c 的 95% CI 对比 ==========")
    print(f"{'方法':<34s} | {'次数':>4s} | {'中位':>5s} | "
          f"{'CI 下':>5s} | {'CI 上':>5s} | {'宽度':>4s}")
    print("-" * 74)
    print(f"{'对级 bootstrap（原版, eval_tracks 同款）':<34s} | "
          f"{len(cs_pair):>4d} | {np.median(cs_pair):>5.0f} | "
          f"{lo_p:>5.0f} | {hi_p:>5.0f} | {hi_p - lo_p:>4.0f}")
    print(f"{'片段级 cluster bootstrap':<34s} | "
          f"{len(cs_seg):>4d} | {np.median(cs_seg):>5.0f} | "
          f"{lo_s:>5.0f} | {hi_s:>5.0f} | {hi_s - lo_s:>4.0f}")
    for name, n, med, lo, hi in rows:
        print(f"{name:<34s} | {n:>4d} | {med:>5.0f} | "
              f"{lo:>5.0f} | {hi:>5.0f} | {hi - lo:>4.0f}")
    print("-" * 74)
    deff = ((hi_s - lo_s) / max(hi_p - lo_p, 1e-9)) ** 2
    print(f"[设计效应] 片段级/对级 CI 宽度比 "
          f"{(hi_s - lo_s) / max(hi_p - lo_p, 1e-9):.1f}× "
          f"⇒ 对级把有效样本量高估约 {deff:.0f} 倍")
    print(f"[结论] c 的诚实不确定度 ≈ ±{(hi_s - lo_s) / 4:.0f} mm/s"
          f"（片段级 95% CI 半宽 {(hi_s - lo_s) / 2:.0f}，"
          f"±1σ≈{(hi_s - lo_s) / 4:.0f}）")

    # ---- 图（新文件名，不覆盖 real_run/ 现有文件）----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(cs_pair.min(), cs_seg.min()) - 5,
                       max(cs_pair.max(), cs_seg.max()) + 5, 60)
    ax.hist(cs_pair, bins=bins, alpha=0.6, density=True,
            label=f"pair-level (orig) CI [{lo_p:.0f},{hi_p:.0f}]")
    ax.hist(cs_seg, bins=bins, alpha=0.6, density=True,
            label=f"segment cluster CI [{lo_s:.0f},{hi_s:.0f}]")
    if cs_blk_main is not None:
        lo_b, hi_b = _ci(cs_blk_main)
        ax.hist(cs_blk_main, bins=bins, alpha=0.45, density=True,
                label=f"time-block (2T) CI [{lo_b:.0f},{hi_b:.0f}]")
    ax.axvline(c2, color="k", lw=1.5, label=f"point estimate {c2:.0f}")
    ax.axvline(C_THEORY, color="g", ls="--", lw=1.2,
               label=f"deep-water theory {C_THEORY:.0f}")
    ax.set_xlabel("c (mm/s)")
    ax.set_ylabel("density")
    ax.set_title("bootstrap distributions of cross-spectral wave speed c")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(OUT, "eval_c_block_bootstrap.png")
    fig.savefig(png, dpi=140)
    print(f"[输出] {png}")


if __name__ == "__main__":
    main()
