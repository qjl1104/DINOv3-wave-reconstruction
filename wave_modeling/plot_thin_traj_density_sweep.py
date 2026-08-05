# wave_modeling/plot_thin_traj_density_sweep.py
"""
轨迹级抽稀扫描汇总图：重建质量 vs 有效物理密度 q=p²（log 横轴）。

读 wave_modeling/real_run/thin_traj_density_sweep_results.csv（驱动逐档追加），
叠加：
  - canonical 基准（q=1.0：975段/主峰率94%/c=1992[1982,2002]/η std 29.00）
  - 上一轮可见性模型（逐帧 Bernoulli 抽稀）已知点，x 取 p_vis²（两视图瞬时
    共见密度，与本图 q 同为"双相机有效观测密度"口径）：
    p_vis=1.0 → q=1.0（同基准）；p_vis=0.5 → q=0.25：54段/60%/c=2007
    [1933,2104]/η std 29.56；p_vis=0.25 → q=0.0625：匹配归零（0 段，× 标记）
  - 开题"覆盖率 1%"目标线：q=0.01 竖虚线

输出：real_run/thin_traj_density_sweep.png

用法：../.venv_fs/Scripts/python.exe plot_thin_traj_density_sweep.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "real_run", "thin_traj_density_sweep_results.csv")
OUT_PNG = os.path.join(HERE, "real_run", "thin_traj_density_sweep.png")

BASE = dict(q=1.0, seg=975, pts=292039, eta=29.00, peak=94,
            c=1992, lo=1982, hi=2002)
# 可见性模型（逐帧抽稀）已知点，x=p_vis²
VIS = [
    dict(q=1.0, seg=975, peak=94, c=1992, lo=1982, hi=2002, eta=29.00),
    dict(q=0.25, seg=54, peak=60, c=2007, lo=1933, hi=2104, eta=29.56),
    dict(q=0.0625, seg=0, peak=None, c=None, lo=None, hi=None, eta=None),
]


def load_rows():
    rows = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(float(r["p"]), int(r["seed"]))] = r  # 重跑后写覆盖
    return list(rows.values())


def fnum(r, k):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return np.nan


def main():
    rows = [r for r in load_rows() if r["status"] == "ok"]
    qs = sorted({float(r["q"]) for r in rows})

    def series(key):
        xs = np.array([float(r["q"]) for r in rows if not np.isnan(fnum(r, key))])
        ys = np.array([fnum(r, key) for r in rows if not np.isnan(fnum(r, key))])
        means = np.array([ys[np.isclose(xs, q)].mean() for q in qs
                          if np.isclose(xs, q).any()])
        qq = np.array([q for q in qs if np.isclose(xs, q).any()])
        return xs, ys, qq, means

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    fig.suptitle("Trajectory-level thinning (physical sparsity) vs visibility "
                 "thinning — x: effective two-view density q")

    def style(ax, ylabel):
        ax.set_xscale("log")
        ax.set_xlabel("effective density q (= p² per-side keep prob; "
                      "visibility pts at p_vis²)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.invert_xaxis()
        ax.axvline(0.01, ls="--", c="purple", lw=1, alpha=0.7)

    def draw(ax, key, base_v, vis_key=None, vis_zero_q=None):
        xs, ys, qq, mm = series(key)
        ax.scatter(xs, ys, c="tab:blue", alpha=0.65, label="traj-level per-seed")
        if len(qq):
            ax.plot(qq, mm, "o-", c="tab:blue", label="traj-level mean")
        ax.scatter([BASE["q"]], [base_v], marker="*", s=190, c="tab:red",
                   zorder=5, label="canonical q=1.0")
        vk = vis_key or key
        vx = [v["q"] for v in VIS if v.get(vk) is not None and v["q"] != 1.0]
        vy = [v[vk] for v in VIS if v.get(vk) is not None and v["q"] != 1.0]
        if vx:
            ax.plot(vx, vy, "s--", c="tab:orange", alpha=0.85,
                    label="visibility model")
        if vis_zero_q is not None:
            ax.scatter([vis_zero_q], [0], marker="x", s=90, c="tab:orange",
                       zorder=6, label="visibility: zero output")
        ax.legend(fontsize=8, loc="best")

    # (a) 3D 段数
    ax = axes[0, 0]
    draw(ax, "n_3d_seg", BASE["seg"], vis_key="seg", vis_zero_q=0.0625)
    style(ax, "3D segments (eval >=30f)")
    ax.set_title("(a) 3D segment count")

    # (b) 主峰率
    ax = axes[0, 1]
    draw(ax, "peak_rate_pct", BASE["peak"], vis_key="peak")
    ax.axhline(94, ls="--", c="tab:red", lw=0.8, alpha=0.5)
    style(ax, "main-peak rate (%)")
    ax.set_title("(b) fraction of >=100f segments peaking at 0.79Hz")

    # (c) c + CI
    ax = axes[1, 0]
    for r in rows:
        c, lo, hi = fnum(r, "c_mm_s"), fnum(r, "ci_lo"), fnum(r, "ci_hi")
        if np.isnan(c):
            continue
        x = float(r["q"])
        if not (np.isnan(lo) or np.isnan(hi)):
            ax.errorbar([x], [c], yerr=[[c - lo], [hi - c]], fmt="none",
                        ecolor="tab:blue", alpha=0.5, capsize=3)
        ax.scatter([x], [c], c="tab:blue", alpha=0.7)
    xs, ys, qq, mm = series("c_mm_s")
    if len(qq):
        ax.plot(qq, mm, "-", c="tab:blue", alpha=0.7)
    ax.errorbar([BASE["q"]], [BASE["c"]],
                yerr=[[BASE["c"] - BASE["lo"]], [BASE["hi"] - BASE["c"]]],
                fmt="*", ms=16, c="tab:red", capsize=4, zorder=5)
    v = VIS[1]
    ax.errorbar([v["q"]], [v["c"]], yerr=[[v["c"] - v["lo"]], [v["hi"] - v["c"]]],
                fmt="s", ms=8, c="tab:orange", capsize=3, alpha=0.85)
    ax.axhline(1993, ls=":", c="k", lw=1, label="linear deep-water @0.783Hz")
    ax.legend(fontsize=8)
    style(ax, "c (mm/s)")
    ax.set_title("(c) wave speed c with segment-level CI")

    # (d) η std
    ax = axes[1, 1]
    draw(ax, "eta_std_mm", BASE["eta"], vis_key="eta")
    ax.axhline(40, ls=":", c="k", lw=1, label="theory amplitude 40mm")
    ax.legend(fontsize=8)
    style(ax, "eta std (mm)")
    ax.set_title("(d) debiased eta std")

    # 开题 1% 目标线标注（只在左上挂文字，四面板共用同一 x）
    axes[0, 0].text(0.011, axes[0, 0].get_ylim()[1] * 0.92,
                    "proposal target: coverage 1%", rotation=90,
                    color="purple", fontsize=8, va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[输出] {OUT_PNG}")
    print(f"CSV 行数 {len(rows)}（ok），q 档：{qs}")


if __name__ == "__main__":
    main()
