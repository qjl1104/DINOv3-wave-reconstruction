# wave_modeling/plot_thin_density_sweep.py
"""
粒子密度抽稀扫描汇总图：重建质量 vs 粒子保留概率（log 横轴）。

读 wave_modeling/real_run/thin_density_sweep_results.csv（驱动脚本逐档追加），
叠加 p=1.0 canonical 基线点（975段/292039点/η std 29.00/主峰率 94%/
c=1992/CI[1982,2002]，2026-08-04 口径修正后基线），输出 2×2 面板：
  (a) 3D 段数  (b) 主峰率  (c) 波速 c 及 honest CI  (d) η std
多 seed 档位画散点 + 均值连线；失效档位（status != ok）在 (a) 面板 x 轴上
画红色 × 标记（性能边界的直接证据）。

用法：../.venv_fs/Scripts/python.exe plot_thin_density_sweep.py
输出：real_run/thin_density_sweep.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "real_run", "thin_density_sweep_results.csv")
OUT_PNG = os.path.join(HERE, "real_run", "thin_density_sweep.png")

# canonical 基线（2026-08-04 口径修正后，AGENTS.md）
BASE = dict(p=1.0, n_3d_seg=975, n_3d_pts=292039, eta_std_mm=29.00,
            peak_rate_pct=94, c_mm_s=1992, ci_lo=1982, ci_hi=2002)


def load_rows():
    rows = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (float(r["p"]), int(r["seed"]))
            rows[key] = r  # 重跑时后写的行覆盖
    return list(rows.values())


def fnum(r, k):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    rows = load_rows()
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    ps = sorted({float(r["p"]) for r in ok})

    def series(key):
        """每密度档的散点列表 + 均值。"""
        xs, ys = [], []
        for r in ok:
            v = fnum(r, key)
            if not np.isnan(v):
                xs.append(float(r["p"]))
                ys.append(v)
        means = [np.mean([y for x, y in zip(xs, ys) if abs(x - p) < 1e-9])
                 for p in ps]
        return np.array(xs), np.array(ys), np.array(ps), np.array(means)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Particle density thinning sweep (v3 chain, foam preserved)")

    def style(ax, ylabel):
        ax.set_xscale("log")
        ax.set_xlabel("keep probability p")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.invert_xaxis()  # 左密右稀 → 沿 +x 方向密度递减，读作"衰减曲线"

    # (a) 3D 段数
    ax = axes[0, 0]
    xs, ys, pm, mm = series("n_3d_seg")
    ax.scatter(xs, ys, c="tab:blue", alpha=0.6, label="per-seed")
    ax.plot(pm, mm, "o-", c="tab:blue", label="mean")
    ax.scatter([BASE["p"]], [BASE["n_3d_seg"]], marker="*", s=180,
               c="tab:red", zorder=5, label="canonical p=1.0")
    for r in bad:
        ax.scatter([float(r["p"])], [ax.get_ylim()[0] + 1], marker="x",
                   c="red", s=60, zorder=6)
    ax.legend(fontsize=8)
    style(ax, "3D segments")
    ax.set_title("(a) 3D segment count")

    # (b) 主峰率
    ax = axes[0, 1]
    xs, ys, pm, mm = series("peak_rate_pct")
    ax.scatter(xs, ys, c="tab:blue", alpha=0.6)
    ax.plot(pm, mm, "o-", c="tab:blue")
    ax.scatter([BASE["p"]], [BASE["peak_rate_pct"]], marker="*", s=180,
               c="tab:red", zorder=5)
    ax.axhline(94, ls="--", c="tab:red", lw=0.8, alpha=0.5)
    style(ax, "main-peak rate (%)")
    ax.set_title("(b) fraction of >=100f segments peaking at 0.79Hz")

    # (c) 波速 c + honest CI
    ax = axes[1, 0]
    for r in ok:
        c, lo, hi = fnum(r, "c_mm_s"), fnum(r, "ci_lo"), fnum(r, "ci_hi")
        if np.isnan(c):
            continue
        x = float(r["p"])
        if not (np.isnan(lo) or np.isnan(hi)):
            ax.errorbar([x], [c], yerr=[[c - lo], [hi - c]], fmt="none",
                        ecolor="tab:blue", alpha=0.5, capsize=3)
        ax.scatter([x], [c], c="tab:blue", alpha=0.7)
    xs, ys, pm, mm = series("c_mm_s")
    ax.plot(pm, mm, "-", c="tab:blue", alpha=0.7)
    ax.errorbar([BASE["p"]], [BASE["c_mm_s"]],
                yerr=[[BASE["c_mm_s"] - BASE["ci_lo"]],
                      [BASE["ci_hi"] - BASE["c_mm_s"]]],
                fmt="*", ms=16, c="tab:red", capsize=4, zorder=5)
    ax.axhline(1993, ls=":", c="k", lw=1, label="linear deep-water @0.783Hz")
    ax.legend(fontsize=8)
    style(ax, "c (mm/s)")
    ax.set_title("(c) wave speed c with segment-level CI")

    # (d) η std
    ax = axes[1, 1]
    xs, ys, pm, mm = series("eta_std_mm")
    ax.scatter(xs, ys, c="tab:blue", alpha=0.6)
    ax.plot(pm, mm, "o-", c="tab:blue")
    ax.scatter([BASE["p"]], [BASE["eta_std_mm"]], marker="*", s=180,
               c="tab:red", zorder=5)
    ax.axhline(40, ls=":", c="k", lw=1, label="theory amplitude 40mm")
    ax.legend(fontsize=8)
    style(ax, "eta std (mm)")
    ax.set_title("(d) debiased eta std")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[输出] {OUT_PNG}")
    n_p = len({float(r['p']) for r in rows})
    print(f"CSV 行数 {len(rows)}（ok {len(ok)}，失效 {len(bad)}，密度档 {n_p}）")


if __name__ == "__main__":
    main()
