# wave_modeling/diag_debias_synthetic.py
"""
任务B：逐片段中位数 debias 的短片段偏置消融诊断。

背景：debias（每片段减自身时间中位数，run_real_pinn._debias /
eval_tracks.evaluate:72-74）只减一个常数。但轨迹 min_len=20~30 帧而波
周期 ≈63 帧（0.797Hz@50fps），不足一周期的正弦片段中位数 ≠ 均值，且
该偏差随片段相位变化——聚合后可能系统性扭曲振幅。

(a) 合成实验：η=40·sin(2π·0.797t)@50fps，随机相位 + 按真实 pkl 统计的
    常值偏置（经验重采样），逐片段中位数 debias，扫片段长度 10..200 帧，
    测幅度衰减率 / 相位误差 / RMS 误差；另做一组按真实片段长度分布混合
    的实验。图上标 min_len=20（PINN）、30（eval）与 63 帧（一周期）。
(b) 真实数据敏感性：min_len ∈ {20,30,63,100} 下跑 eval_tracks 同款流程
    （MAD + debias + η std + FFT 主峰率 + 互谱 c），对比指标变化。

来源说明：不修改 eval_tracks.py / run_real_pinn.py。(b) 的流程主体复制自
eval_tracks.evaluate（仅把 min_len 参数化），底层函数 _regrid_uniform /
_phase_pair / fit_cn / load_series 直接 import 复用。

用法：.venv_fs/Scripts/python.exe wave_modeling/diag_debias_synthetic.py
输出：wave_modeling/diag_debias_synthetic.png
      wave_modeling/diag_debias_synthetic_results.txt
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_tracks import load_series, _regrid_uniform, MIN_SEP, NEAR_SEP  # noqa: E402
from diag_hovmoller_xcorr import F_WAVE, T_WAVE, _phase_pair, fit_cn  # noqa: E402
from run_real_pinn import FPS, C_THEORY  # noqa: E402

PKL = os.path.join(ROOT, "data/trajectories/trajectories_3d_v3nf_hung_dino.pkl")
FIG_OUT = os.path.join(HERE, "diag_debias_synthetic.png")
TXT_OUT = os.path.join(HERE, "diag_debias_synthetic_results.txt")

F_SYN = 0.797                 # 合成波频率 Hz（任务给定）
AMP = 40.0                    # 合成振幅 mm（理论振幅）
T_FRAMES = int(round(FPS / F_SYN))   # 一周期帧数 = 63
MIN_LEN_PINN = 20             # run_real_pinn.load_trajectories 默认
MIN_LEN_EVAL = 30             # eval_tracks.load_series 默认

LINES = []


def log(msg=""):
    print(msg)
    LINES.append(msg)


def _mad_filter(series):
    """与 eval_tracks.evaluate 相同的 MAD 离群剔除（复制以独立调用）。"""
    eta = np.vstack(series)[:, 2]
    med = np.median(eta)
    mad = 1.4826 * np.median(np.abs(eta - med))
    thr = 5 * max(mad, 1e-6)
    series = [s[np.abs(s[:, 2] - med) < thr] for s in series]
    return [s for s in series if len(s) > 0]


def real_bias_and_lens():
    """从真实三维 pkl 统计：(1) debias 量（PCA 系 η 的逐片段中位数，MAD 后、
    debias 前）的经验分布；(2) 片段长度分布（min_len=20，与 pkl 现状一致）。"""
    series = load_series(PKL, min_len=20)
    series = _mad_filter(series)
    lens = np.array([len(s) for s in series])
    biases = np.array([np.median(s[:, 2]) for s in series])
    return biases, lens


# ---------------------------------------------------------------- (a) 合成实验

def _fit_sin(y, t):
    """无截距最小二乘拟合 y = α·sin(ωt) + β·cos(ωt)（debias 后不再含常数项，
    拟合基与真实生成模型一致）。t 对所有片段相同，pinv 只需算一次。"""
    w = 2 * np.pi * F_SYN
    G = np.c_[np.sin(w * t), np.cos(w * t)]
    M = np.linalg.pinv(G)              # (2, L)
    coef = y @ M.T                     # (N, 2)
    amp = np.hypot(coef[:, 0], coef[:, 1])
    phase = np.arctan2(coef[:, 1], coef[:, 0])
    return amp, phase


def synth_at_len(L, biases, rng, n_seg=4000, noise_std=0.0):
    """固定片段长度 L 的 Monte Carlo：返回 (聚合等效振幅比, 拟合振幅比中位,
    相位误差圆 std[rad], RMS 误差[mm], |中位数偏差|中位[mm])。"""
    phi = rng.uniform(0, 2 * np.pi, n_seg)
    b = rng.choice(biases, size=n_seg, replace=True)
    t = np.arange(L) / FPS
    y0 = AMP * np.sin(2 * np.pi * F_SYN * t[None, :] + phi[:, None])
    y = y0 + b[:, None]
    if noise_std > 0:
        y = y + rng.normal(0, noise_std, y.shape)
    m = np.median(y, axis=1)           # 真实 pipeline 的 debias 量
    yd = y - m[:, None]
    # 聚合等效振幅（对应 eval_tracks 的 η std 口径：std·√2）
    amp_pooled = np.sqrt(2.0) * yd.std() / AMP
    # 逐片段正弦拟合（已知频率）
    amp_fit, ph_fit = _fit_sin(yd, t)
    ph_err = (ph_fit - phi + np.pi) % (2 * np.pi) - np.pi
    ph_std = np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * ph_err))) + 1e-12))
    # debias 误差 = 真实零均值信号被减掉的量：m - b = median(y0)（+噪声影响）
    err = m - b
    rms = float(np.sqrt(np.mean(err ** 2)))
    return (amp_pooled, float(np.median(amp_fit) / AMP), ph_std, rms,
            float(np.median(np.abs(err))))


def synth_mixture(biases, lens, rng):
    """按真实片段长度分布混合的实验（每条长度真实出现一次）。"""
    n = len(lens)
    phi = rng.uniform(0, 2 * np.pi, n)
    b = rng.choice(biases, size=n, replace=True)
    deb_all, true_all = [], []
    for L, p, bb in zip(lens, phi, b):
        t = np.arange(L) / FPS
        y0 = AMP * np.sin(2 * np.pi * F_SYN * t + p)
        y = y0 + bb
        deb_all.append(y - np.median(y))
        true_all.append(y0)
    deb_all = np.concatenate(deb_all)
    true_all = np.concatenate(true_all)
    amp_pooled = np.sqrt(2.0) * deb_all.std() / AMP
    rms = float(np.sqrt(np.mean((deb_all - true_all) ** 2)))
    return amp_pooled, rms


# ---------------------------------------------------------------- (b) 真实敏感性

def evaluate_minlen(min_len):
    """eval_tracks.evaluate 的 min_len 参数化复制版（流程与口径保持一致，
    返回数字而非只打印）。"""
    series = load_series(PKL, min_len=min_len)
    lens = np.array([len(s) for s in series])
    series = _mad_filter(series)
    for s in series:                       # 逐片段中位数 debias（同 evaluate）
        s[:, 2] -= np.median(s[:, 2])
    allpts = np.vstack(series)
    eta_std = float(allpts[:, 2].std())

    # FFT 主峰率（≥100 帧片段；同 evaluate）
    pk_f = []
    for s in series:
        fr = np.round(s[:, 3] * FPS).astype(int)
        order = np.argsort(fr)
        fr, et = fr[order], s[order, 2]
        _, uniq = np.unique(fr, return_index=True)
        fr, et = _regrid_uniform(fr[uniq], et[uniq])
        if len(fr) < 100:
            continue
        e = et - et.mean()
        sp = np.abs(np.fft.rfft(e * np.hanning(len(e))))
        fq = np.fft.rfftfreq(len(e), 1 / FPS)
        pk_f.append(fq[np.argmax(sp[1:]) + 1])
    pk_f = np.array(pk_f)
    peak_frac = float(np.mean(np.abs(pk_f - F_WAVE) < 0.1)) if len(pk_f) else np.nan

    # 互谱相位测速（同 evaluate，含带通与两遍去模糊）
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
    pk_bp = []
    for fr, et, _ in frags:
        if len(fr) < 100:
            continue
        e = et - et.mean()
        sp = np.abs(np.fft.rfft(e * np.hanning(len(e)), 8192))
        fq = np.fft.rfftfreq(8192, 1 / FPS)
        band = (fq >= 0.5) & (fq <= 1.2)
        pk_bp.append(fq[band][np.argmax(sp[band])])
    f_meas = float(np.median(pk_bp)) if pk_bp else F_WAVE
    raw = []
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
            raw.append((tau, du, dv, amp_min * np.sqrt(n_win), np.hypot(du, dv)))
    c_val = c_lo = c_hi = np.nan
    if len(raw) >= 6:
        near = [p for p in raw if p[4] < NEAR_SEP]
        c1, n1, _ = fit_cn(near if len(near) >= 3 else raw)
        unwrapped = []
        for tau, du, dv, w, sep in raw:
            pred = (du * n1[0] + dv * n1[1]) / c1
            cands = [tau + k * T_WAVE for k in range(-3, 4)]
            unwrapped.append((min(cands, key=lambda x: abs(x - pred)), du, dv, w))
        c2, _, _ = fit_cn(unwrapped)
        rng = np.random.default_rng(1)
        cs = np.array([fit_cn([unwrapped[k] for k in
                               rng.integers(len(unwrapped), size=len(unwrapped))])[0]
                       for _ in range(200)])
        c_val, c_lo, c_hi = c2, np.percentile(cs, 2.5), np.percentile(cs, 97.5)
    return dict(min_len=min_len, n_seg=len(series), n_pts=len(allpts),
                len_med=float(np.median(lens)), eta_std=eta_std,
                n_fft=len(pk_f), peak_frac=peak_frac, n_pairs=len(raw),
                c=c_val, c_lo=c_lo, c_hi=c_hi)


# ---------------------------------------------------------------- main

def main():
    rng = np.random.default_rng(42)
    log(f"[数据] {os.path.basename(PKL)}")
    biases, lens = real_bias_and_lens()
    log(f"[数据] 片段 {len(lens)} 条（≥20帧，MAD后）| 长度 中位 {np.median(lens):.0f} "
        f"p10 {np.percentile(lens, 10):.0f} p90 {np.percentile(lens, 90):.0f} "
        f"max {lens.max()}")
    log(f"[数据] 真实 debias 量（逐片段η中位数）：中位 {np.median(biases):.1f} mm | "
        f"p5 {np.percentile(biases, 5):.1f} p95 {np.percentile(biases, 95):.1f} | "
        f"|b| 中位 {np.median(np.abs(biases)):.1f} mm")
    log(f"[数据] ≥{T_FRAMES}帧(一周期) 片段 {(lens >= T_FRAMES).sum()} 条 "
        f"({(lens >= T_FRAMES).mean():.0%})；<63 帧 {(lens < T_FRAMES).sum()} 条")

    # ---- (a1) 长度扫描
    log("\n=== (a) 合成扫描：η=40·sin(2π·0.797t)，随机相位+真实偏置，逐片段中位数 debias ===")
    log("（注：常值偏置 b 在中位数中精确抵消——median(y+b)=median(y)+b——")
    log(" 故衰减与偏置大小无关，误差全部来自短窗正弦中位数≠均值）")
    sweep_L = list(range(10, 201))
    res = {L: synth_at_len(L, biases, rng,
                           n_seg=4000 if L <= 100 else 2000) for L in sweep_L}
    amp_pooled = np.array([res[L][0] for L in sweep_L])
    amp_fit = np.array([res[L][1] for L in sweep_L])
    ph_std = np.degrees([res[L][2] for L in sweep_L])
    rms_err = np.array([res[L][3] for L in sweep_L])
    log(f"{'L(帧)':>6} {'聚合振幅比':>10} {'拟合振幅比':>10} {'相位std(°)':>10} "
        f"{'RMS误差mm':>10} {'|中位偏|mm':>10}")
    for L in [10, 15, 20, 25, 30, 40, 50, 63, 80, 100, 126, 150, 200]:
        a, af, ps, rm, md = res[L]
        log(f"{L:>6} {a:>10.3f} {af:>10.3f} {np.degrees(ps):>10.2f} "
            f"{rm:>10.2f} {md:>10.2f}")

    # ---- (a2) 真实长度分布混合
    mix_amp, mix_rms = synth_mixture(biases, lens, rng)
    log(f"\n[a-混合] 按真实长度分布（中位 {np.median(lens):.0f} 帧）切段："
        f"聚合等效振幅 = {mix_amp * AMP:.1f} mm（衰减 {(1 - mix_amp):.1%}），"
        f"RMS 误差 {mix_rms:.2f} mm")

    # ---- (a3) 噪声鲁棒性（σ=3mm 跟踪噪声）
    log("\n[a-噪声] σ=3mm 逐点噪声下关键长度对比（聚合振幅比 / RMS误差mm）：")
    for L in [20, 30, 63, 100]:
        a, _, _, rm, _ = synth_at_len(L, biases, rng, n_seg=4000, noise_std=3.0)
        log(f"  L={L:>3}: {a:.3f} / {rm:.2f}（无噪声: {res[L][0]:.3f} / {res[L][3]:.2f}）")

    # ---- (a) 图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax in axes:
        for x, lab in [(MIN_LEN_PINN, "min_len=20 (PINN)"),
                       (MIN_LEN_EVAL, "min_len=30 (eval)"),
                       (T_FRAMES, "63 = 1 period")]:
            ax.axvline(x, ls="--", lw=1.2,
                       color={"min_len=20 (PINN)": "tab:red",
                              "min_len=30 (eval)": "tab:orange",
                              "63 = 1 period": "tab:green"}[lab])
        ax.set_xlabel("segment length L (frames @50fps)")
    axes[0].plot(sweep_L, amp_pooled, label="pooled std·√2 (eval metric)")
    axes[0].plot(sweep_L, amp_fit, label="per-seg sine fit (median)")
    axes[0].axhline(mix_amp, ls=":", color="tab:purple",
                    label=f"real length mix = {mix_amp:.3f}")
    axes[0].set_ylabel("amplitude ratio vs true 40mm")
    axes[0].set_title("(a1) amplitude attenuation after debias")
    axes[0].legend(fontsize=8)
    axes[1].plot(sweep_L, ph_std)
    axes[1].set_ylabel("phase error circ-std (deg)")
    axes[1].set_title("(a2) phase error of per-seg sine fit")
    axes[2].plot(sweep_L, rms_err)
    axes[2].set_ylabel("RMS error (mm)")
    axes[2].set_title("(a3) RMS of subtracted median bias")
    for ax, x in [(axes[0], MIN_LEN_PINN)]:
        pass
    # 文本标注（避免与线重叠，统一放顶部）
    for ax in axes:
        ylim = ax.get_ylim()
        ax.text(MIN_LEN_PINN + 1, ylim[1], "20/30", color="tab:red",
                fontsize=8, va="top")
        ax.text(T_FRAMES + 1, ylim[1], "63", color="tab:green",
                fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    log(f"\n[图] {FIG_OUT}")

    # ---- (b) 真实数据敏感性
    log("\n=== (b) 真实数据敏感性：min_len 对 debias 后指标的影响 ===")
    log(f"{'min_len':>8} {'片段数':>7} {'总点数':>8} {'η std mm':>9} "
        f"{'FFT段数':>8} {'主峰率':>7} {'配对数':>7} {'c mm/s':>16}")
    rows = []
    for ml in [20, 30, 63, 100]:
        r = evaluate_minlen(ml)
        rows.append(r)
        c_str = (f"{r['c']:.0f} [{r['c_lo']:.0f},{r['c_hi']:.0f}]"
                 if np.isfinite(r['c']) else "n/a")
        log(f"{r['min_len']:>8} {r['n_seg']:>7} {r['n_pts']:>8} "
            f"{r['eta_std']:>9.2f} {r['n_fft']:>8} "
            f"{r['peak_frac']:>7.0%} {r['n_pairs']:>7} {c_str:>16}")

    # ---- 结论
    a20, a30, a63 = res[MIN_LEN_PINN][0], res[MIN_LEN_EVAL][0], res[T_FRAMES][0]
    n_long = int((lens >= T_FRAMES).sum())
    pt_frac_long = lens[lens >= T_FRAMES].sum() / lens.sum()
    r20, r63, r100 = rows[0], rows[2], rows[3]
    log("\n=== 结论 ===")
    log("(a) 短片段【个体】层面：debias 显著扭曲振幅——")
    log(f"    L=20 聚合等效振幅只剩 {a20:.1%}（衰减 {(1 - a20):.1%}），"
        f"L=30 剩 {a30:.1%}（衰减 {(1 - a30):.1%}），相位误差圆 std "
        f"{np.degrees(res[MIN_LEN_PINN][2]):.0f}°/{np.degrees(res[MIN_LEN_EVAL][2]):.0f}°；"
        f"L≥63（一周期）衰减 <0.1%。")
    log("    机制：不足一周期的窗口内正弦中位数≠均值，减中位数把段内真实")
    log("    波面起伏当作常值偏置抹掉一部分（与偏置 b 大小无关，b 精确抵消）。")
    log(f"(a) 聚合层面：按真实长度分布混合，衰减仅 {(1 - mix_amp):.1%}——"
        f"pkl 点数由长片段主导（≥63帧 {n_long} 条贡献 {pt_frac_long:.0%} 的点），"
        f"短片段在 pooled η std 中权重很小。")
    log(f"(b) 真实数据验证：min_len 20→63→100，η std "
        f"{r20['eta_std']:.2f}→{r63['eta_std']:.2f}→{r100['eta_std']:.2f} mm，"
        f"c {r20['c']:.0f}→{r63['c']:.0f}→{r100['c']:.0f} mm/s，"
        f"主峰率 {r20['peak_frac']:.0%}→{r63['peak_frac']:.0%}→{r100['peak_frac']:.0%}"
        f"——基线指标对 min_len 几乎不敏感。")
    log("回答：1) 当前 min_len 下 debias 对基线聚合振幅（η std 口径）的扭曲 "
        f"≈{(1 - mix_amp):.1%}，可忽略；η std 29mm vs 理论 40mm 的缺口"
        f"不能归咎于 debias；")
    log("      2) 但短片段个体的 η 被严重压缩（L=20 衰减 44%），逐片段使用短段"
        "（PINN 训练点、逐段振幅统计）会引入低振幅偏差；")
    log("      3) 不建议仅为修正 debias 把 min_len 提到 63——基线指标无实质"
        f"变化却要损失 {(lens < T_FRAMES).sum()} 条（{(lens < T_FRAMES).mean():.0%}）"
        "片段；若做逐段振幅/相位分析，只用 ≥63 帧段即可。")

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\n[存] {TXT_OUT}")


if __name__ == "__main__":
    main()
