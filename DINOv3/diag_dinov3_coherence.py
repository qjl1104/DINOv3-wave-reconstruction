# DINOv3/diag_dinov3_coherence.py
"""
DINOv3 逐帧点云的时空相干性诊断（判决能否并入 PINN 训练集）。

原理与 wave_modeling/diag_hovmoller_xcorr.py 相同，但对象换成"网格节点"：
DINOv3 逐帧独立匹配、无粒子跟踪，把每帧点云投进 (u,v) 规则网格，
每个网格节点的 η(t) 就是一条"虚拟浪高仪"时间序列。若逐帧匹配正确，
节点序列应含 0.79 Hz 主峰，且节点间 0.79 Hz 相位差与沿传播方向的距离
呈线性（斜率 1/c）。反之若匹配是噪声，节点序列无主峰、相位无结构。

网格节点时间跨度 20s（1000 帧@50Hz），远比粒子片段（0.6–3s）长，
因此这个检验对 DINOv3 点云是相当严格的。

用法：../.venv_fs/Scripts/python.exe diag_dinov3_coherence.py
输出：diag_coherence.png + 打印指标
"""

import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "wave_modeling"))
from diag_hovmoller_xcorr import F_WAVE, T_WAVE, fit_cn, _phase_pair  # noqa: E402

PKL = (sys.argv[1] if len(sys.argv) > 1 else
       os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "pointclouds_1000f.pkl"))
OUT = os.path.dirname(os.path.abspath(__file__))
FPS = 50.0
C_THEORY = 1976.0      # mm/s（0.79 Hz 深水规则波）
CELL = 250.0           # 网格边长 mm
MIN_COVER = 0.25       # 节点至少覆盖 25% 的帧
MIN_SEP = 300.0
NEAR_SEP = 1000.0


def load_clouds():
    with open(PKL, "rb") as f:
        clouds = pickle.load(f)
    frames = sorted(clouds.keys())
    return frames, clouds


def pca_eta(frames, clouds):
    """全部点 PCA 主平面 → 每帧点集的 (u, v, eta)。"""
    allpts = np.vstack([clouds[f] for f in frames if len(clouds[f])])
    c0 = allpts.mean(axis=0)
    _, _, vt = np.linalg.svd(allpts - c0, full_matrices=False)
    e_u, e_v, n = vt[0], vt[1], vt[2]
    out = {}
    for f in frames:
        p = clouds[f]
        if len(p) == 0:
            continue
        d = p - c0
        out[f] = np.c_[d @ e_u, d @ e_v, d @ n]
    return out, (c0, n)


def grid_nodes(frames, uv):
    """(u,v) 规则网格：节点时间序列 = 该格内点的逐帧 η 均值。"""
    alluv = np.vstack(list(uv.values()))
    u0, u1 = np.percentile(alluv[:, 0], [1, 99])
    v0, v1 = np.percentile(alluv[:, 1], [1, 99])
    # 逐帧聚合
    series = {}
    for f, pts in uv.items():
        iu_f = np.clip(((pts[:, 0] - u0) / CELL).astype(int), 0, 98)
        iv_f = np.clip(((pts[:, 1] - v0) / CELL).astype(int), 0, 98)
        key = iu_f * 100 + iv_f
        for k in np.unique(key):
            m = key == k
            node = (iu_f[m][0], iv_f[m][0])
            series.setdefault(node, []).append((f, np.median(pts[m, 2])))
    nodes = {}
    n_frames = len(frames)
    for node, lst in series.items():
        if len(lst) >= MIN_COVER * n_frames:
            fr = np.array([x[0] for x in lst])
            et = np.array([x[1] for x in lst])
            # 节点位置 = 格中心
            pos = (u0 + (node[0] + 0.5) * CELL, v0 + (node[1] + 0.5) * CELL)
            # MAD 剔除节点序列内的离群帧（坏匹配帧）
            med = np.median(et)
            mad = 1.4826 * np.median(np.abs(et - med))
            keep = np.abs(et - med) < 5 * max(mad, 1e-6)
            nodes[node] = (fr[keep], et[keep], pos)
    return nodes, (u0, v0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames, clouds = load_clouds()
    npts = np.array([len(clouds[f]) for f in frames])
    print(f"[数据] {len(frames)} 帧 | 每帧点数中位 {np.median(npts):.0f} "
          f"（{npts.min()}–{npts.max()}）| 空帧 {np.sum(npts == 0)}")

    uv, _ = pca_eta(frames, clouds)
    nodes, _ = grid_nodes(frames, uv)
    print(f"[网格] {CELL:.0f}mm 网格，覆盖率≥{MIN_COVER:.0%} 的节点 {len(nodes)} 个")

    # ---- 每节点 FFT：0.79Hz 是否为主峰 ----
    pk_f, pk_a, amp079 = [], [], []
    for fr, et, _ in nodes.values():
        if len(fr) < 200:
            continue
        e = et - et.mean()
        sp = np.abs(np.fft.rfft(e * np.hanning(len(e))))
        fq = np.fft.rfftfreq(len(e), 1 / FPS)
        k = np.argmax(sp[1:]) + 1
        pk_f.append(fq[k])
        pk_a.append(2 * sp[k] / len(e))
        k079 = int(round(F_WAVE * len(e) / FPS))
        amp079.append(2 * sp[k079] / len(e))
    pk_f, pk_a = np.array(pk_f), np.array(pk_a)
    frac_079 = np.mean(np.abs(pk_f - F_WAVE) < 0.1)
    print(f"[FFT] ≥200 样本节点 {len(pk_f)} 个 | 主峰落在 0.79±0.1Hz 的比例 "
          f"{frac_079:.0%} | 主峰频率中位 {np.median(pk_f):.3f} Hz | "
          f"0.79Hz 振幅中位 {np.median(amp079):.1f} mm")

    # ---- 互谱相位法测 c（节点对，全程重叠）----
    keys = list(nodes.keys())
    raw = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            fr_i, et_i, pi = nodes[keys[i]]
            fr_j, et_j, pj = nodes[keys[j]]
            du = pj[0] - pi[0]
            dv = pj[1] - pi[1]
            sep = np.hypot(du, dv)
            if sep < MIN_SEP:
                continue
            out = _phase_pair(fr_i, et_i, fr_j, et_j)
            if out is None:
                continue
            tau, amp_min, n_win = out
            raw.append((tau, du, dv, amp_min * np.sqrt(n_win), sep, amp_min))
    print(f"[phase] 有效节点对 {len(raw)} 个")
    if len(raw) < 6:
        print("[phase] 有效对太少，无法拟合 c。")
        return

    near = [p for p in raw if p[4] < NEAR_SEP]
    if len(near) >= 3:
        c1, n1, _ = fit_cn(near)
    else:
        c1, n1, _ = fit_cn(raw)
    unwrapped = []
    for tau, du, dv, w, sep, r in raw:
        pred = (du * n1[0] + dv * n1[1]) / c1
        cands = [tau + k * T_WAVE for k in range(-3, 4)]
        unwrapped.append((min(cands, key=lambda x: abs(x - pred)), du, dv, w))
    c2, n2, res = fit_cn(unwrapped)
    theta = np.degrees(np.arctan2(n2[1], n2[0]))
    rms_ms = np.sqrt(np.mean(res ** 2)) * 1000
    print(f"[phase] c = {c2:.0f} mm/s = {c2 / 1000:.3f} m/s | "
          f"方向角 {theta:.1f}° | 时滞残差 RMS {rms_ms:.0f} ms | "
          f"理论 {C_THEORY:.0f} mm/s，偏差 {abs(c2 - C_THEORY) / C_THEORY:.1%}")

    rng = np.random.default_rng(1)
    cs = np.array([fit_cn([unwrapped[k] for k in
                           rng.integers(len(unwrapped), size=len(unwrapped))])[0]
                   for _ in range(200)])
    print(f"[phase] bootstrap 95% CI: [{np.percentile(cs, 2.5):.0f}, "
          f"{np.percentile(cs, 97.5):.0f}] mm/s")

    # ---- 图：节点分布+方向 / τ-距离散点 / FFT 主峰分布 / 沿传播方向 Hovmöller ----
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.5))
    pos = np.array([v[2] for v in nodes.values()])
    ax[0].scatter(pos[:, 0] / 1000, pos[:, 1] / 1000, s=40)
    ax[0].quiver(pos[:, 0].mean() / 1000, pos[:, 1].mean() / 1000,
                 n2[0], n2[1], angles="xy", scale_units="xy", scale=0.5,
                 color="r", width=0.008)
    ax[0].set_title(f"grid nodes & fitted dir {theta:.0f} deg")
    ax[0].set_xlabel("u (m)"); ax[0].set_ylabel("v (m)"); ax[0].set_aspect("equal")
    proj = np.array([(p[1] * n2[0] + p[2] * n2[1]) / 1000 for p in unwrapped])
    taus = np.array([p[0] for p in unwrapped])
    ax[1].scatter(proj, taus, s=25, alpha=0.7)
    xs = np.linspace(proj.min(), proj.max(), 10)
    ax[1].plot(xs, xs * 1000 / c2, "r-", label=f"fit c={c2:.0f} mm/s")
    ax[1].plot(xs, xs / (C_THEORY / 1000), "g--", label=f"theory {C_THEORY:.0f}")
    ax[1].set_xlabel("projected separation (m)"); ax[1].set_ylabel("lag tau (s)")
    ax[1].legend(); ax[1].set_title("lag vs separation")
    if len(pk_f):
        ax[2].hist(pk_f, bins=25, alpha=0.7)
        ax[2].axvline(F_WAVE, color="r", ls="--", label="0.79 Hz (paper)")
        ax[2].set_xlabel("node FFT peak frequency (Hz)")
        ax[2].set_ylabel("node count")
        ax[2].legend(); ax[2].set_title(f"peak-freq dist ({frac_079:.0%} at 0.79)")
    # Hovmöller along propagation: 节点按投影位置分箱
    projs = (pos @ n2)
    pbins = np.linspace(projs.min(), projs.max(), 14)
    H = np.full((len(pbins) - 1, len(frames)), np.nan)
    fidx = {f: k for k, f in enumerate(frames)}
    for node, (fr, et, p) in nodes.items():
        pr = np.dot(np.array(p), n2)
        b = np.clip(np.digitize(pr, pbins) - 1, 0, len(pbins) - 2)
        row = H[b]
        col = np.array([fidx[f] for f in fr])
        # 同箱多节点：取均值
        cur = row[col]
        both = ~np.isnan(cur)
        row[col] = np.where(both, (cur + et) / 2, et)
    vmax = np.nanstd(H) * 2
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("white")
    im = ax[3].imshow(H, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                      extent=[0, len(frames) / FPS, pbins[-1] / 1000, pbins[0] / 1000],
                      origin="upper")
    ax[3].set_xlabel("t (s)"); ax[3].set_ylabel("position along wave dir (m)")
    ax[3].set_title("Hovmoller along propagation (expect diagonal)")
    fig.colorbar(im, ax=ax[3])
    fig.tight_layout()
    stem = os.path.splitext(os.path.basename(PKL))[0]
    png = os.path.join(OUT, f"diag_coherence_{stem}.png")
    fig.savefig(png, dpi=130)
    print(f"[输出] {png}")


if __name__ == "__main__":
    main()
