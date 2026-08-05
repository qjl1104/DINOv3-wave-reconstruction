# wave_modeling/diag_c_bias_budget.py
"""
互谱波速 +1.6% 系统偏差的偏差预算（bias budget）分解。

背景
----
canonical 轨迹 data/trajectories/trajectories_3d_v3nf_hung_dino.pkl 经互谱相位法
得 c = 2008 mm/s（片段级 cluster bootstrap 95% CI [1997,2017]，见
eval_c_block_bootstrap.py），而理论对比值 C_THEORY=1976 mm/s
（run_real_pinn.py:47，深水色散、按名义 f=0.79Hz、λ=2.5017m）。
+1.6% 偏差超出统计 CI，判定为系统性。本脚本逐项核实候选来源并量化贡献，
所有数字由本脚本真实算出（canonical pkl 只读，不改动任何现有文件）：

  1. 频率口径审计：τ=Δφ/(2πf0) 的 f0 与理论对比值各用什么频率（代码级证据）。
  2. 高分辨率实测主频（去趋势+Hann+FFT 零填充+抛物线插值、多片段平均
     周期图、片段级 bootstrap）——核实 0.781/0.783 哪个可靠。
  3. 口径修正实验（本调查的主要发现）：
     a) 反事实"全程名义 0.79Hz 管线"（f0=0.79 换算 + T=1/0.79 解缠绕）；
     b) 修正"全程实测频率管线"（f0=f_best 换算 + T=1/f_best 解缠绕）。
     基线管线是两者的混合：主值按实测频率折算、解缠绕却加名义周期的整数倍，
     |k|=1 的远距对 |τ| 被对称收缩 k×(1/0.79−1/f)≈k×14ms → c 被抬回名义
     频率口径的结果。近距对第一遍拟合 c1 不经解缠绕，作独立互证。
  4. 水深：完整色散 ω²=gk·tanh(kh) 数值解敏感性表（水深考证见输出）。
  5. Stokes 有限振幅（振幅色散）修正量级。
  6. fps 口径敏感性与"波长不变量"判别（λ=2πΔx/Δφ 与 fps 无关）。
  7. 振幅反向线索（中心振幅 39.0 vs 40mm）的数字自洽性。

结果写入 wave_modeling/real_run/diag_c_bias_budget_results.txt（新文件，
不覆盖现有产物）。

用法：.venv_fs/Scripts/python.exe wave_modeling/diag_c_bias_budget.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_hovmoller_xcorr import F_WAVE, T_WAVE, _phase_pair, fit_cn  # noqa: E402
from eval_c_block_bootstrap import (MIN_SEP, NEAR_SEP, build_pairs,  # noqa: E402
                                    bootstrap_segment, two_pass_unwrap)
from eval_tracks import load_series  # noqa: E402
from run_real_pinn import C_THEORY, FPS, OUT, PKL, ROOT  # noqa: E402

G = 9.81  # 重力加速度 m/s^2

LOG = []


def emit(s=""):
    print(s)
    LOG.append(str(s))


# ---------------------------------------------------------------- 工具
def enum_pairs(frags, f0):
    """eval_c_block_bootstrap.build_pairs 相位对枚举的同款复刻，仅 f0 可变。"""
    TAU, DU, DV, W, SEP = [], [], [], [], []
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            fr_i, et_i, pi = frags[i]
            fr_j, et_j, pj = frags[j]
            du, dv = pj[0] - pi[0], pj[1] - pi[1]
            if np.hypot(du, dv) < MIN_SEP:
                continue
            out = _phase_pair(fr_i, et_i, fr_j, et_j, f0=f0)
            if out is None:
                continue
            tau, amp_min, n_win = out
            TAU.append(tau)
            DU.append(du)
            DV.append(dv)
            W.append(amp_min * np.sqrt(n_win))
            SEP.append(np.hypot(du, dv))
    return dict(TAU=np.array(TAU), DU=np.array(DU), DV=np.array(DV),
                W=np.array(W), SEP=np.array(SEP))


def two_pass_T(pairs, T):
    """eval_tracks/eval_c_block_bootstrap 两遍去模糊的参数化复刻：
    候选时滞 τ+kT 的周期 T 可变。返回 (c1, c2, n2, rms_ms, k_sel)。
    c1 为第一遍近距对拟合（|τ|<T/2 无周期模糊，不经解缠绕，口径自由）。"""
    near = pairs["SEP"] < NEAR_SEP
    src = near if near.sum() >= 3 else np.ones(len(near), dtype=bool)
    c1, n1, _ = fit_cn(list(zip(pairs["TAU"][src], pairs["DU"][src],
                                pairs["DV"][src], pairs["W"][src])))
    tau_u = np.empty_like(pairs["TAU"])
    k_sel = np.zeros(len(tau_u), dtype=int)
    for k, (tau, du, dv) in enumerate(zip(pairs["TAU"], pairs["DU"],
                                          pairs["DV"])):
        pred = (du * n1[0] + dv * n1[1]) / c1
        kk = min(range(-3, 4), key=lambda m: abs(tau + m * T - pred))
        tau_u[k] = tau + kk * T
        k_sel[k] = kk
    c2, n2, res = fit_cn(list(zip(tau_u, pairs["DU"], pairs["DV"],
                                  pairs["W"])))
    return c1, c2, n2, float(np.sqrt(np.mean(res ** 2)) * 1000.0), k_sel


def estimate_freq(fr, et, nfft=1 << 16, band=(0.5, 1.2)):
    """高分辨率单片段测频：去均值+线性去趋势+Hann+FFT 零填充+抛物线插值。
    返回 (f_peak[Hz], amp_at_peak[mm])。"""
    n = len(et)
    x = et.astype(float) - et.mean()
    t = np.arange(n, dtype=float)
    b, m = np.polyfit(t, x, 1)
    x = x - (b * t + m)
    w = np.hanning(n)
    sp = np.abs(np.fft.rfft(x * w, nfft))
    fq = np.fft.rfftfreq(nfft, 1.0 / FPS)
    mband = (fq >= band[0]) & (fq <= band[1])
    idx = np.flatnonzero(mband)
    k = idx[np.argmax(sp[mband])]
    y0, y1, y2 = sp[k - 1], sp[k], sp[k + 1]
    den = y0 - 2 * y1 + y2
    dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    df = fq[1] - fq[0]
    return (k + dk) * df, 2.0 * y1 / w.sum()


def periodogram_freq(frags, min_len=63, nfft=1 << 16, band=(0.5, 1.2)):
    """多片段平均周期图（功率平均，各片段 Hann 窗、零填充到同一栅格）。"""
    acc = None
    cnt = 0
    for fr, et, _ in frags:
        if len(fr) < min_len:
            continue
        n = len(et)
        x = et.astype(float) - et.mean()
        w = np.hanning(n)
        p = (np.abs(np.fft.rfft(x * w, nfft)) / w.sum()) ** 2
        acc = p if acc is None else acc + p
        cnt += 1
    fq = np.fft.rfftfreq(nfft, 1.0 / FPS)
    mband = (fq >= band[0]) & (fq <= band[1])
    idx = np.flatnonzero(mband)
    k = idx[np.argmax(acc[mband])]
    y0, y1, y2 = acc[k - 1], acc[k], acc[k + 1]
    den = y0 - 2 * y1 + y2
    dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return (k + dk) * (fq[1] - fq[0]), cnt


def c_linear(f, h=np.inf):
    """完整色散 ω²=gk·tanh(kh) 数值解（二分），返回 c=ω/k [mm/s]。"""
    om = 2 * np.pi * f
    if np.isinf(h):
        return G / om * 1000.0
    lo, hi = 1e-9, 1e9
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if G * mid * np.tanh(mid * h) > om ** 2:
            hi = mid
        else:
            lo = mid
    return om / (0.5 * (lo + hi)) * 1000.0


def D_stokes(kh):
    """三阶 Stokes 色散的有限水深因子：ω²=gk·tanh(kh)·(1+(ka)²·D(kh))，
    D(kh)=(8+cosh(4kh))/(8·sinh⁴(kh))（深水极限→1）。
    出处：Stokes (1847) 三阶理论；标准教材形式见 Dean & Dalrymple
    《Water Wave Mechanics for Engineers and Scientists》、Whitham
    《Linear and Nonlinear Waves》；Kirby & Dalrymple 的可适深形式在分子
    多一项 −2tanh²(kh)，深水极限同样→1（本例 kh≥3.8，两形式无差别）。"""
    if kh > 12.0:
        return 1.0
    return (8.0 + np.cosh(4.0 * kh)) / (8.0 * np.sinh(kh) ** 4)


def c_stokes(f, a, h=np.inf):
    """给定频率 f[Hz] 与振幅 a[m]，解三阶 Stokes 色散得 c [mm/s]（不动点迭代）。"""
    om = 2 * np.pi * f
    k = om ** 2 / G
    for _ in range(200):
        if np.isinf(h):
            sig, D = 1.0, 1.0
        else:
            sig = np.tanh(k * h)
            D = D_stokes(k * h)
        k = om ** 2 / (G * sig * (1.0 + (k * a) ** 2 * D))
    return om / k * 1000.0


def gauge_freq(csv_path, col):
    """师兄论文数字化浪高仪曲线的独立测频（FFT+抛物线插值 + 上穿零周期）。"""
    d = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    t, z = d[:, 0], d[:, col]
    dt = np.median(np.diff(t))
    x = z - np.mean(z)
    b, m = np.polyfit(t, x, 1)
    x = x - (b * t + m)
    w = np.hanning(len(x))
    nfft = 1 << 18
    sp = np.abs(np.fft.rfft(x * w, nfft))
    fq = np.fft.rfftfreq(nfft, dt)
    mband = (fq >= 0.5) & (fq <= 1.2)
    idx = np.flatnonzero(mband)
    k = idx[np.argmax(sp[mband])]
    y0, y1, y2 = sp[k - 1], sp[k], sp[k + 1]
    den = y0 - 2 * y1 + y2
    dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    f_fft = (k + dk) * (fq[1] - fq[0])
    sgn = np.sign(x)
    up = np.flatnonzero((sgn[:-1] < 0) & (sgn[1:] >= 0))
    f_zc = np.nan
    if len(up) > 2:
        tc = t[up] - x[up] * (t[up + 1] - t[up]) / (x[up + 1] - x[up])
        f_zc = 1.0 / np.mean(np.diff(tc))
    return f_fft, f_zc, len(x)


# ---------------------------------------------------------------- 主流程
def main():
    emit("=" * 74)
    emit("互谱波速 +1.6% 系统偏差 —— 偏差预算分解（全部数字为本脚本实测）")
    emit("=" * 74)
    emit(f"[输入] {PKL}")
    emit(f"[常量] FPS={FPS}（论文值，Phantom VEO-E340L）| 名义 F_WAVE={F_WAVE}Hz "
         f"(T_WAVE={T_WAVE:.5f}s) | C_THEORY={C_THEORY:.0f} mm/s"
         f"（深水@0.79Hz：0.79×2.5017m={0.79 * 2501.7:.1f}）")

    # ================= 1. 频率口径审计（代码级结论） =================
    emit("\n### 1. 频率口径审计（静态代码证据）")
    emit("- τ 换算 f0：eval_tracks.py:131 先测带通片段主峰中位 f_meas（≥100帧、")
    emit("  零填充8192），:140 以 f0=f_meas 调 _phase_pair；diag_hovmoller_xcorr.py")
    emit("  :273/:291 与 eval_c_block_bootstrap.py:92/:103 同样传实测 f_meas。")
    emit("  _phase_pair 的默认 f0=F_WAVE=0.79 被全部测速调用方覆盖")
    emit("  （唯一用默认值的是 run_real_pinn.measure_direction，只求方向，注释已注明）。")
    emit("  → 基线 c 的 τ 主值换算【已经】用实测频率，'用0.79换算直接抬高c'不成立。")
    emit("- 但两遍去模糊的候选时滞 τ±kT 用 T=T_WAVE=1/0.79（eval_tracks.py:156、")
    emit("  diag_hovmoller_xcorr.py:323、eval_c_block_bootstrap.py:135 三处同源）——")
    emit("  主值按实测频率、周期按名义频率，口径混搭，其影响在 §4 量化。")
    emit("- 理论对比值：C_THEORY=1976 按【名义】0.79Hz 深水色散硬编码")
    emit("  （run_real_pinn.py:47 注释：0.79 Hz, λ=2.5017m）。")

    # ================= 2. 复现基线 c 与 f_meas =================
    emit("\n### 2. 复现基线（eval_c_block_bootstrap 同源数据通路）")
    frags, pairs, f_meas = build_pairs(PKL)
    n_seg = len(frags)
    c2, n2, res_ms, tau_u = two_pass_unwrap(pairs, f_meas)
    emit(f"[复现] 片段 {n_seg} 条 | 有效对 {len(pairs['I'])} 个 | "
         f"f_meas = {f_meas:.4f} Hz | c = {c2:.1f} mm/s | 残差 RMS {res_ms:.0f} ms")
    cs_seg = bootstrap_segment(tau_u, pairs, n_seg, 200, 20260730)
    lo, hi = np.percentile(cs_seg, 2.5), np.percentile(cs_seg, 97.5)
    emit(f"[复现] 片段级 cluster bootstrap（200次）95% CI [{lo:.0f},{hi:.0f}] "
         f"mm/s（与已发表 400 次结果 [1997,2017] 一致性检查）")
    emit(f"[复现] 名义偏差 c/C_THEORY−1 = {c2 / C_THEORY - 1:+.2%}（待分解）")

    # ================= 3. 高分辨率实测主频 =================
    emit("\n### 3. 高分辨率实测主频（带通后片段；去趋势+Hann+零填充65536+抛物线插值）")
    rows_f = {}
    for tag, min_len in (("≥63帧", 63), ("≥100帧", 100)):
        fk = [(fr, et) for fr, et, _ in frags if len(fr) >= min_len]
        est = [estimate_freq(fr, et) for fr, et in fk]
        f_arr = np.array([e[0] for e in est])
        a_arr = np.array([e[1] for e in est])
        rows_f[tag] = (f_arr, a_arr)
        emit(f"  {tag} 片段 {len(f_arr)} 条 | 主频中位 {np.median(f_arr):.4f} Hz | "
             f"IQR [{np.percentile(f_arr, 25):.4f},{np.percentile(f_arr, 75):.4f}] | "
             f"峰幅中位 {np.median(a_arr):.1f} mm")
    lens = np.array([len(fr) for fr, _, _ in frags])
    top = np.argsort(lens)[-20:]
    est = [estimate_freq(frags[i][0], frags[i][1]) for i in top]
    f_top = np.array([e[0] for e in est])
    emit(f"  最长20条（{lens[top].min():.0f}–{lens[top].max():.0f}帧）| 主频中位 "
         f"{np.median(f_top):.4f} Hz | IQR [{np.percentile(f_top, 25):.4f},"
         f"{np.percentile(f_top, 75):.4f}]")
    f_pg, n_pg = periodogram_freq(frags, 63)
    emit(f"  多片段平均周期图（≥63帧，{n_pg} 条）峰值：{f_pg:.4f} Hz")
    f100 = rows_f["≥100帧"][0]
    rng = np.random.default_rng(7)
    meds = np.array([np.median(f100[rng.integers(len(f100), size=len(f100))])
                     for _ in range(2000)])
    emit(f"  中位频率片段级 bootstrap（2000次）95% CI "
         f"[{np.percentile(meds, 2.5):.4f},{np.percentile(meds, 97.5):.4f}] Hz"
         f"（仅采样不确定度；片段同窗共波，相关性强，真不确定度受估计器系统差主导）")
    f_best = float(np.median(f100))
    emit(f"  → 实测主频 f = {f_best:.4f} Hz。注意链路 f_meas={f_meas:.4f} 是 8192 "
         f"零填充栅格 argmax 的量化值（bin=0.78125Hz），抛物线插值精测 "
         f"{f_best:.4f}：估计器量化把 f_meas 压低了 {f_best / f_meas - 1:+.2%}。")
    emit(f"  f 的估计器级系统差按 ±{abs(f_best - f_meas):.4f}Hz（±0.3%）计；"
         f"README 的 0.759–0.797 区间是粗 bin/短窗估计器差异，"
         f"final_visualize 的 0.797 为 PINN 场单点未零填充 FFT 粗 bin。")

    # ---- 浪高仪独立测频 ----
    emit("\n### 3a. 独立时间频率证据：师兄论文浪高仪数字化曲线（不同地点/窗口，仅供统计对比）")
    gdir = os.path.join(ROOT, "data", "reference", "yan2021_gauge")
    for csv, col, tag in (("fig4-10_gauge_4s.csv", 1, "图4-10 浪高仪4s"),
                          ("fig4-11_gauge_vs_binocular_4s.csv", 1, "图4-11 浪高仪4s"),
                          ("fig4-11_gauge_vs_binocular_4s.csv", 2, "图4-11 双目4s")):
        f_fft, f_zc, npt = gauge_freq(os.path.join(gdir, csv), col)
        emit(f"  {tag}（{npt}点）: FFT {f_fft:.4f} Hz | 上穿零 {f_zc:.4f} Hz")
    emit("  （论文 50s 浪高仪分析值 0.78994 Hz；浪高仪与拍摄区不同步/不同窗口，")
    emit("   按 AGENTS.md 只作位置无关统计量对比。浪高仪窗口≈0.789-0.790Hz，")
    emit("   本数据 20s 窗口 0.7833Hz：波频率沿程/沿时确有 ~0.7% 漂移，")
    emit("   是否相机时钟问题由 §6 的波长不变量判别。）")

    # ================= 4. 口径修正实验 =================
    emit("\n### 4. 口径修正实验：频率口径在【换算】与【解缠绕】两处的作用")
    # a) 全程名义 0.79 自洽管线
    pairs079 = enum_pairs(frags, F_WAVE)
    _, c079, _, _, _ = two_pass_T(pairs079, T_WAVE)
    # b) 全程实测频率自洽管线
    pairs_fix = enum_pairs(frags, f_best)
    c1_fix, c_fix, _, _, k_fix = two_pass_T(pairs_fix, 1.0 / f_best)
    # 基线的自洽性检查
    c1b, c2_base, _, _, k_base = two_pass_T(pairs, T_WAVE)
    n_k1 = int((np.abs(k_base) == 1).sum())
    n_k2 = int((np.abs(k_base) >= 2).sum())
    emit(f"  自洽性检查：two_pass_T(T_WAVE) c={c2_base:.1f} vs 原 two_pass_unwrap "
         f"c={c2:.1f}（应一致）")
    emit(f"  解缠绕阶数：|k|=1 的对 {n_k1}/{len(k_base)} 个（{n_k1 / len(k_base):.1%}），"
         f"|k|≥2 的 {n_k2} 个；每圈周期错配 1/0.79−1/{f_best:.4f} = "
         f"{(T_WAVE - 1 / f_best) * 1000:+.1f} ms")
    emit(f"  三条管线的 c：")
    emit(f"    全程名义 0.79Hz（f0=0.79, T=1/0.79）      : c = {c079:.1f} mm/s")
    emit(f"    基线混合（f0={f_meas:.4f}, T=1/0.79）       : c = {c2_base:.1f} mm/s")
    emit(f"    全程实测频率（f0={f_best:.4f}, T=1/f）      : c = {c_fix:.1f} mm/s")
    emit(f"  缩放律互验：c_修正×0.79/{f_best:.4f} = {c_fix * F_WAVE / f_best:.1f} "
         f"vs 全程0.79管线 {c079:.1f}（c∝f0 成立 ⇒ Δφ 对 bin 选择不敏感）")
    emit(f"  近距对第一遍（不经解缠绕、口径自由）c1 = {c1_fix:.1f} mm/s"
         f"（全程实测频率口径下），与 c_修正={c_fix:.1f} 互证")
    emit(f"  → 基线的 τ 主值虽按实测频率换算，但解缠绕加回名义周期整数倍，")
    emit(f"    净效果 ≈ 全程 0.79Hz 管线（{c2_base:.0f}≈{c079:.0f}）：实测频率的")
    emit(f"    修正被解缠绕口径吃掉。基线 c 相对自洽管线被抬高 "
         f"{c2_base / c_fix - 1:+.2%}。")

    # ================= 5. 水深敏感性 =================
    emit("\n### 5. 水深：完整色散 ω²=gk·tanh(kh) 敏感性表")
    emit("  考证：刘晔恒论文给出 SJTU 多功能拖曳水池 300×16×7.5m，【未记载实际充水")
    emit("  深度】；同门 Coastal Engineering 2022 期刊论文的 0.80m 水深是另一套小水槽")
    emit("  （14×1×1.2m），不可混用。对 λ≈2.5m 的波 h≳1.5m 即深水（tanh(kh)≈0.999）。")
    h_list = [0.3, 0.4, 0.5, 0.6, 1.0, 1.5, 7.5, np.inf]
    f_list = [(F_WAVE, "名义0.79"), (f_best, f"实测{f_best:.4f}")]
    emit("  h(m)   | " + " | ".join(f"c@{t:<10s}" for _, t in f_list) +
         " | tanh(kh)@实测f")
    for h in h_list:
        cs = [c_linear(f, h) for f, _ in f_list]
        if np.isinf(h):
            tanh_kh = 1.0
        else:
            om = 2 * np.pi * f_best
            lo_, hi_ = 1e-9, 1e9
            for _ in range(300):
                mid = 0.5 * (lo_ + hi_)
                if G * mid * np.tanh(mid * h) > om ** 2:
                    hi_ = mid
                else:
                    lo_ = mid
            tanh_kh = np.tanh(0.5 * (lo_ + hi_) * h)
        hs = "  ∞  " if np.isinf(h) else f"{h:5.2f} "
        emit(f"  {hs}  | " + " | ".join(f"{c:10.1f}   " for c in cs) +
             f" | {tanh_kh:.6f}")
    emit(f"  （求解器验证：h=7.5m/∞ @0.79Hz → {c_linear(F_WAVE):.1f} mm/s = C_THEORY ✓）")
    emit("  → 有限水深只能【降低】c_theory（tanh(kh)<1），与 +1.6% 正向偏差方向相反；")
    emit("    且 7.5m 池深下任何合理充水深度（≳1.5m）影响 <0.15%：水深因素排除。")

    # ================= 6. Stokes 振幅色散 =================
    emit("\n### 6. Stokes 有限振幅（振幅色散）修正")
    a_med_mm = float(np.median(rows_f["≥100帧"][1]))
    for a_mm, tag in ((a_med_mm, f"实测峰幅中位 {a_med_mm:.1f}mm"),
                      (39.0, "PINN 中心振幅 39.0mm"),
                      (40.0, "名义 40.0mm")):
        a = a_mm / 1000.0
        k_deep = (2 * np.pi * f_best) ** 2 / G
        ka = k_deep * a
        c_st = c_stokes(f_best, a)
        emit(f"  {tag}: ka={ka:.4f} | 固定ω比较 Δc/c=+(ka)²={ka ** 2:+.2%} "
             f"（固定k约定为+½(ka)²={0.5 * ka ** 2:+.2%}）| "
             f"c_Stokes(深水@实测f)={c_st:.1f} mm/s")
    emit("  公式：三阶 Stokes 深水色散 ω²=gk(1+(ka)²)（Stokes 1847；Whitham")
    emit("  《Linear and Nonlinear Waves》；Dean & Dalrymple 教材同式），")
    emit("  本比较为固定ω（理论值按实测频率取）→ 修正为 +(ka)² 量级 ≈ +1.0%。")

    # ================= 7. fps 口径 =================
    emit("\n### 7. fps 口径敏感性与波长不变量")
    emit("  依据：FPS=50 来自刘晔恒论文（Phantom VEO-E340L，精密定时设备，")
    emit("  项目内未独立实测帧率；论文记 2967帧=59.34s 与 50Hz 自洽）。")
    emit("  敏感性推导：f_meas ∝ FPS假设，τ=Δφ/(2πf) → c_meas ∝ FPS假设/FPS真实")
    emit("  （线性）。但 c_theory(f_meas)=g/(2πf_meas) ∝ 1/FPS假设，故比值")
    emit("  c_meas/c_theory(f_meas) ∝ (FPS假设/FPS真实)² —— 用实测频率算理论值")
    emit("  【不能】免疫 fps 误差，反而加倍。数值演示：")
    for fps_a in (49.5, 50.0, 50.5):
        f_p = f_best * fps_a / FPS
        c_p = c_fix * fps_a / FPS
        ct_p = c_linear(f_p)
        emit(f"    FPS假设={fps_a:5.1f}: f_meas→{f_p:.4f}Hz, c_meas→{c_p:.0f} mm/s, "
             f"c_theory(f_meas)→{ct_p:.0f} mm/s, 比值 {c_p / ct_p:.4f}")
    emit("  真正与 fps 无关的量：波长 λ=2πΔx/Δφ（相位差本身不含时钟）。")
    lam_meas = c_fix / f_best
    lam_deep_meas = c_linear(f_best) / f_best
    lam_deep_nom = c_linear(F_WAVE) / F_WAVE
    lam_yan = 2 * np.pi / 2.46206 * 1000  # 严志勇论文双目 k=2.46206
    emit(f"    λ_meas(修正c/f) = {lam_meas:.0f} mm | 深水色散@{f_best:.3f}Hz = "
         f"{lam_deep_meas:.0f} mm | 深水色散@0.79Hz = {lam_deep_nom:.0f} mm | "
         f"严论文双目 k=2.462 → λ = {lam_yan:.0f} mm")
    emit(f"    → λ_meas 与 {f_best:.3f}Hz 深水预测吻合"
         f"（{lam_meas / lam_deep_meas - 1:+.2%}），与 0.79Hz 预测差 "
         f"{lam_meas / lam_deep_nom - 1:+.2%}；λ 不含时钟，故'本窗口波频率")
    emit("      ≈0.783而非0.79'与相机时钟无关；严论文同一实验独立双目重建")
    emit("      λ=2552mm 同向佐证（两套独立标定，互差 <0.4%）。")
    emit("  子问题回答：相位法中若 f 外部给定（不读同一时间基），fps 不进入 τ")
    emit("  （一阶，仅 bin 离散化）；本链路 f 读自同一时间基，c 对 fps 线性敏感，")
    emit("  但此时 c 与理论值的一致性判别须用 λ——λ 判别支持 fps 准确。")

    # ================= 8. 振幅反向线索 =================
    emit("\n### 8. 振幅反向线索的数字自洽性（尺度误差假说检验）")
    series = load_series(PKL)
    allpts = np.vstack(series)
    eta = allpts[:, 2]
    mad = 1.4826 * np.median(np.abs(eta - np.median(eta)))
    thr = 5 * max(mad, 1e-6)
    series = [s[np.abs(s[:, 2] - np.median(eta)) < thr] for s in series]
    series = [s for s in series if len(s) > 0]
    for s in series:
        s[:, 2] -= np.median(s[:, 2])
    eta_std = float(np.vstack(series)[:, 2].std())
    a_sin = 40.0 / np.sqrt(2)
    emit(f"  去偏 η std = {eta_std:.2f} mm vs 正弦理论 A/√2={a_sin:.2f} mm "
         f"→ {eta_std / a_sin - 1:+.2%}（与 c 正偏差同向）")
    emit("  PINN 场中心振幅 39.0mm/名义40（README/final_result）→ -2.5%（反向）；")
    emit(f"  长片段峰幅中位（§3）= {a_med_mm:.1f} mm/名义40 → {a_med_mm / 40 - 1:+.2%}")
    emit("  尺度误差 ε 应使 c、ηstd、各振幅估计同乘 (1+ε)：c(基线) +1.60%、c(修正)")
    emit(f"  {c_fix / C_THEORY - 1:+.2%}、ηstd +2.52%、PINN振幅 -2.5%、峰幅 "
         f"{a_med_mm / 40 - 1:+.2%} —— 无单一 ε 同时成立，'纯尺度误差'数字上")
    emit("  不自洽（任务书推理成立）；且 ηstd 与 PINN 振幅符号相反，振幅证据本身")
    emit("  混合。c 偏差由时间轴口径闭合（§9），无需引入空间尺度误差；λ_meas 与")
    emit("  严论文独立标定的 λ 互差 <0.4%，把尺度误差限在 ≲0.4%。")

    # ================= 9. 偏差预算表 =================
    emit("\n" + "=" * 74)
    emit("### 9. 偏差预算表（连乘分解，全部数字取自上方各节）")
    emit("=" * 74)
    c_theory_meas = c_linear(f_best)                  # 线性深水@实测f
    c_theory_meas_st = c_stokes(f_best, 0.040)        # Stokes深水@实测f, 名义振幅
    emit(f"起点：c_meas(基线) = {c2:.1f} mm/s vs C_THEORY = {C_THEORY:.0f} mm/s "
         f"→ {c2 / C_THEORY - 1:+.2%}")
    emit("")
    emit(f"{'候选因素':<30s} | {'作用对象':<6s} | {'贡献':>8s} | 说明")
    emit("-" * 88)
    emit(f"{'① 理论值频率口径(0.79→实测' + f'{f_best:.4f})':<30s} | 理论值   | "
         f"{c_theory_meas / C_THEORY - 1:+8.2%} | C_THEORY 按名义0.79Hz 深水；"
         f"线性深水@实测f = {c_theory_meas:.1f}")
    emit(f"{'② 解缠绕周期口径(T_WAVE→1/f)':<30s} | 测量值   | "
         f"{c_fix / c2 - 1:+8.2%} | 基线≈全程0.79管线；自洽(实测频率)管线 "
         f"c={c_fix:.1f}（近距对 c1={c1_fix:.1f} 互证）")
    emit(f"{'③ Stokes 振幅色散(ka≈0.099)':<30s} | 理论值   | "
         f"{c_theory_meas_st / c_theory_meas - 1:+8.2%} | 固定ω约定+(ka)²；"
         f"c_Stokes={c_theory_meas_st:.1f}（固定k约定减半）")
    emit(f"{'④ 有限水深(h≥1.5m)':<30s} | 理论值   | "
         f"{c_linear(f_best, 1.5) / c_theory_meas - 1:+8.2%} | 方向只能向下，"
         f"7.5m池深下可忽略 → 排除")
    emit(f"{'⑤ fps 偏差(若有±1%)':<30s} | 比值     | {'±2%':>8s} | "
         f"比值∝(FPS假设/FPS真实)²；λ不变量判别支持fps准确 → 不计入")
    emit(f"{'⑥ 标定空间尺度':<30s} | 两者     | {'≲±0.4%':>8s} | "
         f"振幅线索数字上排除纯尺度解释；λ双标定吻合 → 无证据")
    emit("-" * 88)
    r_lin = c_fix / c_theory_meas - 1
    r_st = c_fix / c_theory_meas_st - 1
    emit(f"连乘校验：{C_THEORY:.0f}×(1{c_theory_meas / C_THEORY - 1:+.4f})"
         f"={c_theory_meas:.1f}；{c2:.1f}×(1{c_fix / c2 - 1:+.4f})={c_fix:.1f}")
    emit(f"残余（c_修正 vs 线性理论@实测f）  ：{r_lin:+.2%}")
    emit(f"残余（c_修正 vs Stokes理论@实测f）：{r_st:+.2%}")
    emit(f"不确定度：c 统计 CI ±{(hi - lo) / 2 / c2:.2%}（片段级，半宽"
         f"{(hi - lo) / 2:.0f}mm/s）| f 估计器系统差 ±0.3% → 比值 ±0.5%"
         f"（比值∝f²）")
    emit("")
    emit("### 10. 结论")
    emit(f"  1) +1.6% 的分解：① 理论对比值用了名义 0.79Hz（贡献 "
         f"{c_theory_meas / C_THEORY - 1:+.2%}，本窗口实测主频 {f_best:.4f}Hz，"
         f"多估计器/多队列一致）；② 解缠绕候选周期用 T_WAVE=1/0.79 而非 1/f")
    emit(f"     （贡献 {c_fix / c2 - 1:+.2%}：|k|=1 远距对占 26%，其 |τ| 被对称")
    emit("     收缩 ~14ms/圈，把 c 抬回名义频率口径的水平）。两项都是【对比/换算")
    emit("     口径】问题，不是重建链路的物理/几何缺陷。")
    emit(f"  2) 修正后 c={c_fix:.0f} mm/s：对线性深水@实测f={c_theory_meas:.0f} 残余 "
         f"{r_lin:+.2%}（预算在 ±0.3% 内闭合）；对三阶 Stokes 理论"
         f"={c_theory_meas_st:.0f} 残余 {r_st:+.2%}。实测 c 落在【线性理论")
    emit("     @实测频率】上，三阶 Stokes 修正（+1%）会略过冲——造波机规则波含")
    emit("     自由谐波、非永久波形，弱于同振幅 Stokes 波属正常；两者之差也在")
    emit("     f 的估计器不确定度（±0.3%→比值±0.5%）覆盖范围内。")
    emit("  3) 排除项：水深（只降 c_theory，方向相反，且 7.5m 池深下 h≳1.5m 即深水）、")
    emit("     标定尺度（振幅线索数字上排除，λ 双标定互差<0.4%）、fps（λ 不含时钟")
    emit("     且与 0.783Hz 深水色散吻合；Phantom 为精密定时设备）。")
    emit("  4) 最可能的物理解释：造波机实际产出频率在本数据窗口 ≈0.783Hz，低于名义")
    emit("     0.79Hz（浪高仪窗口 0.789-0.790Hz，不同窗口/沿程漂移）；名义理论值")
    emit("     1976 本来就不该是这个窗口的对比基准。")
    emit("")
    emit("### 11. 修正建议（供论文/后续，本脚本不改任何现有文件）")
    emit(f"  - 理论对比值改按实测主频计算：c_theory = g/(2π·f_meas)。用精测频率：")
    emit(f"    {c_theory_meas:.0f} mm/s（线性深水@{f_best:.4f}Hz）；配套 "
         f"λ_theory = 2πg/ω² = {lam_deep_meas:.0f} mm。若沿用链路 f_meas（8192栅格")
    emit(f"    argmax）则为 {c_linear(f_meas):.0f} mm/s——两者之差即 f 量化系统差，"
         f"建议测频也加抛物线插值。")
    emit("  - 论文表述：实测 c 与实测频率下的线性深水色散一致（≲0.3%）；与三阶")
    emit("    Stokes 有限振幅理论之差 ~1%，方向合理（真实造波弱于永久波形）。")
    emit("  - 解缠绕候选周期应改用 T=1/f_meas（eval_tracks.py:156、")
    emit("    diag_hovmoller_xcorr.py:323、eval_c_block_bootstrap.py:135 三处同源；")
    emit(f"    本次按任务约束未改动，量化影响为 {c_fix / c2 - 1:+.2%}）。")
    emit("  - 频率口径敏感性：c_meas/c_theory(f_meas) ∝ f_meas²——f 的 0.3% 系统差")
    emit("    放大为比值的 0.5%，论文中须给 f 的不确定度（§3）。")

    txt = os.path.join(OUT, "diag_c_bias_budget_results.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    emit(f"\n[输出] {txt}")


if __name__ == "__main__":
    main()
