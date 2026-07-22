# wave_modeling/eval_tracks.py
"""
3D 轨迹集统一评估器：任意 [frame,X,Y,Z] 片段 pkl → 相干性指标。
用于对比：粒子链 v2 / DINOv3 描述子跟踪 / 调参 KLT，同一杆秤。

指标：
- 数据量：片段数、总点数、片段长度分布
- 去偏质量：MAD 离群剔除 + 逐片段中位数 debias 后 η std（≈波幅为好）
- 频率锚点：≥100 帧片段 FFT 主峰落在 0.79±0.1Hz 的比例、频率中位
- 相位测速：互谱相位法 c（对角化两遍及 200 次 bootstrap CI）
用法：.venv_fs/Scripts/python.exe wave_modeling/eval_tracks.py <pkl> [标签]
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_hovmoller_xcorr import F_WAVE, T_WAVE, _phase_pair, fit_cn  # noqa: E402
from run_real_pinn import FPS, C_THEORY  # noqa: E402

MIN_SEP = 300.0
NEAR_SEP = 1000.0


def load_series(pkl_path, min_len=30):
    with open(pkl_path, "rb") as f:
        trajs = pickle.load(f)
    trajs = [np.asarray(t, dtype=np.float64) for t in trajs if len(t) >= min_len]
    allpts = np.vstack([t[:, 1:4] for t in trajs])
    c0 = allpts.mean(axis=0)
    _, _, vt = np.linalg.svd(allpts - c0, full_matrices=False)
    series = []
    for t in trajs:
        d = t[:, 1:4] - c0
        series.append(np.c_[d @ vt[0], d @ vt[1], d @ vt[2], t[:, 0] / FPS])
    return series


def evaluate(pkl_path, label):
    series = load_series(pkl_path)
    lens = np.array([len(s) for s in series])
    allpts = np.vstack(series)
    # MAD 离群剔除
    eta = allpts[:, 2]
    mad = 1.4826 * np.median(np.abs(eta - np.median(eta)))
    thr = 5 * max(mad, 1e-6)
    series = [s[np.abs(s[:, 2] - np.median(eta)) < thr] for s in series]
    # 逐片段 debias
    for s in series:
        s[:, 2] -= np.median(s[:, 2])
    allpts = np.vstack(series)
    print(f"\n=== {label} ===")
    print(f"[量] 片段 {len(series)} 条（≥30 帧）| 总点 {len(allpts)} | "
          f"片段长 中位 {np.median(lens):.0f} p90 {np.percentile(lens, 90):.0f} "
          f"max {lens.max()}")
    print(f"[质] 去偏后 η std = {allpts[:, 2].std():.2f} mm（波幅 ~40mm 为锚）")

    pk_f = []
    for s in series:
        if len(s) < 100:
            continue
        e = s[:, 2] - s[:, 2].mean()
        sp = np.abs(np.fft.rfft(e * np.hanning(len(e))))
        fq = np.fft.rfftfreq(len(e), 1 / FPS)
        pk_f.append(fq[np.argmax(sp[1:]) + 1])
    pk_f = np.array(pk_f)
    if len(pk_f):
        frac = np.mean(np.abs(pk_f - F_WAVE) < 0.1)
        print(f"[频] ≥100帧片段 {len(pk_f)} 条 | 主峰 0.79±0.1Hz 比例 {frac:.0%} | "
              f"频率中位 {np.median(pk_f):.3f} Hz")

    # 相位测速
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
            if np.hypot(du, dv) < MIN_SEP:
                continue
            out = _phase_pair(fr_i, et_i, fr_j, et_j)
            if out is None:
                continue
            tau, amp_min, n_win = out
            raw.append((tau, du, dv, amp_min * np.sqrt(n_win),
                        np.hypot(du, dv), amp_min))
    print(f"[相] 有效片段对 {len(raw)} 个（0.79Hz 幅值≥4mm）")
    if len(raw) >= 6:
        near = [p for p in raw if p[4] < NEAR_SEP]
        c1, n1, _ = fit_cn(near if len(near) >= 3 else raw)
        unwrapped = []
        for tau, du, dv, w, sep, r in raw:
            pred = (du * n1[0] + dv * n1[1]) / c1
            cands = [tau + k * T_WAVE for k in range(-3, 4)]
            unwrapped.append((min(cands, key=lambda x: abs(x - pred)), du, dv, w))
        c2, n2, res = fit_cn(unwrapped)
        rng = np.random.default_rng(1)
        cs = np.array([fit_cn([unwrapped[k] for k in
                               rng.integers(len(unwrapped), size=len(unwrapped))])[0]
                       for _ in range(200)])
        print(f"[相] c = {c2:.0f} mm/s（理论 {C_THEORY:.0f}，偏差 "
              f"{abs(c2 - C_THEORY) / C_THEORY:.1%}）| 残差 RMS "
              f"{np.sqrt(np.mean(res ** 2)) * 1000:.0f} ms | 95% CI "
              f"[{np.percentile(cs, 2.5):.0f}, {np.percentile(cs, 97.5):.0f}]")
    else:
        print("[相] 有效对不足，无法测 c")


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
