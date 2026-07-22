# wave_modeling/final_visualize.py
"""
最终成果可视化：加载 run_real_pinn.py 训练的模型（pinn_real.pt，含传播
坐标系旋转 rot）与轨迹数据，产出：
- final_result.png：数据 vs 重建波面（中值时刻）、稠密 Hovmöller(ξ,t)
  叠加数据点、时间均方根振幅图、代表点时间序列对比
- final_field.npz：稠密时空波面场 η(ξ,ζ,t)（论文用图/后续分析）
- 打印重建场提取的波浪参数（主频、波长、振幅）与理论值对比

用法：.venv_fs/Scripts/python.exe wave_modeling/final_visualize.py [traj_pkl]
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_v2 import PINNWaveV2  # noqa: E402
from run_real_pinn import C_THEORY, FPS, OUT, prepare_data  # noqa: E402

F_WAVE_PAPER = 0.79   # 论文规则波频率 Hz

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_SPAN_SNAPSHOTS = 4   # 快照个数（跨一个周期）


def load_model():
    ck = torch.load(os.path.join(OUT, "pinn_real.pt"), map_location="cpu",
                    weights_only=False)
    model = PINNWaveV2(ck["bounds"], c_init=C_THEORY, learn_c=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model.to(DEVICE), ck


def predict_grid(model, xi, zeta, ts):
    """ξ(Nx,) ζ(Ny,) ts(Nt,) → η (Nx, Ny, Nt) mm，分批推理。"""
    g = np.meshgrid(xi, zeta, ts, indexing="ij")
    q = torch.tensor(np.c_[g[0].ravel(), g[1].ravel(), g[2].ravel()],
                     dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(q), 8192):
            out.append(model.predict(q[i:i + 8192].to(DEVICE)).cpu())
    return torch.cat(out).numpy().reshape(len(xi), len(zeta), len(ts))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj_pkl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(OUT), "..", "data/trajectories/trajectories_3d_v2_dino.pkl")
    model, ck = load_model()
    rot = ck.get("rot", np.eye(2))
    series, allpts, _ = prepare_data(traj_pkl)
    allpts = allpts.copy()
    allpts[:, :2] = allpts[:, :2] @ rot.T   # 与训练一致的传播坐标系
    b = ck["bounds"]
    xi = np.linspace(b["x"][0], b["x"][1], 120)
    zeta = np.linspace(b["y"][0], b["y"][1], 80)
    ts = np.linspace(b["t"][0], b["t"][1], 200)
    print(f"[重建] 网格 {len(xi)}×{len(zeta)}×{len(ts)}")
    eta_g = predict_grid(model, xi, zeta, ts)

    # ---- 波成分提取：PINN 在数据稀疏/边缘区会生成慢变结构（~0.1Hz，
    # 全场 rms 可达 100mm），而数据本身 std 仅 40mm。波信号在 0.79Hz，
    # 对每个网格点的时间序列做 [0.6,1.0]Hz 带通，得到真实波成分场。
    dt = ts[1] - ts[0]
    E = np.fft.rfft(eta_g - eta_g.mean(axis=2, keepdims=True), axis=2)
    fq_g = np.fft.rfftfreq(len(ts), dt)
    E[:, :, (fq_g < 0.6) | (fq_g > 1.0)] = 0.0
    eta_w = np.fft.irfft(E, n=len(ts), axis=2)

    # 数据覆盖掩模：无数据点（任一时刻）400mm 内的网格视为外推区，不展示
    from scipy.spatial import cKDTree
    tree = cKDTree(allpts[:, :2])
    gx, gz = np.meshgrid(xi, zeta, indexing="ij")
    dist, _ = tree.query(np.c_[gx.ravel(), gz.ravel()])
    cov = (dist.reshape(gx.shape) < 400.0)

    # ---- 从波成分场提取波浪参数 ----
    cx, cy = len(xi) // 2, len(zeta) // 2
    sig = eta_w[cx, cy]
    sp = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    fq = np.fft.rfftfreq(len(sig), dt)
    f_dom = fq[np.argmax(sp[1:]) + 1]
    amp_ts = 2 * sp[np.argmax(sp[1:]) + 1] / len(sig)
    mid = len(ts) // 2
    sigx = eta_w[:, cy, mid] - eta_w[:, cy, mid].mean()
    spx = np.abs(np.fft.rfft(sigx * np.hanning(len(sigx))))
    kx = np.fft.rfftfreq(len(sigx), xi[1] - xi[0])
    lam = 1.0 / kx[np.argmax(spx[1:]) + 1]
    print(f"[参数] 波成分场：主频 {f_dom:.3f} Hz（理论 {F_WAVE_PAPER}）| "
          f"中心点振幅 {amp_ts:.1f} mm（理论 40）| 波长 {lam:.0f} mm（理论 2502）| "
          f"原始场 rms {eta_g.std():.1f} mm（含慢变/边缘伪结构）| "
          f"波成分 rms {eta_w[cov].std():.1f} mm（覆盖区内）")

    # ---- 图 ----
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    vlim = 3 * allpts[:, 2].std()
    wlim = 2.5 * eta_w[cov].std()
    t_mid = 0.5 * sum(b["t"])

    # (0,0) 全帧数据散点（展示真实空间覆盖）
    sc = ax[0, 0].scatter(allpts[:, 0] / 1000, allpts[:, 1] / 1000,
                          c=allpts[:, 2], cmap="RdBu_r", s=3,
                          vmin=-vlim, vmax=vlim)
    ax[0, 0].set_title(f"data, all {len(allpts)} pts (mm)")
    fig.colorbar(sc, ax=ax[0, 0])

    # (0,1) 原始 PINN 场（覆盖掩模外留白）
    raw_show = np.where(cov, eta_g[:, :, mid], np.nan)
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("white")
    im = ax[0, 1].pcolormesh(xi / 1000, zeta / 1000, raw_show.T,
                             cmap=cmap, shading="auto", vmin=-vlim, vmax=vlim)
    ax[0, 1].set_title(f"PINN raw field at t={t_mid:.1f}s, masked (mm)")
    fig.colorbar(im, ax=ax[0, 1])

    # (0,2) 波成分场（带通 0.6-1.0Hz，覆盖掩模）
    wave_show = np.where(cov, eta_w[:, :, mid], np.nan)
    im = ax[0, 2].pcolormesh(xi / 1000, zeta / 1000, wave_show.T,
                             cmap=cmap, shading="auto", vmin=-wlim, vmax=wlim)
    ax[0, 2].set_title(f"wave component (0.6-1.0Hz) at t={t_mid:.1f}s (mm)")
    fig.colorbar(im, ax=ax[0, 2])
    for a in ax[0, :]:
        a.set_xlabel("xi (m)"); a.set_ylabel("zeta (m)")

    # (1,0) 波成分 Hovmöller（ζ 中值带）+ 数据点叠加
    z_med = np.median(allpts[:, 1])
    iz = np.argmin(np.abs(zeta - z_med))
    hm = eta_w[:, max(iz - 1, 0):iz + 2, :].mean(axis=1)
    im = ax[1, 0].pcolormesh(ts, xi / 1000, hm, cmap=cmap, shading="auto",
                             vmin=-wlim, vmax=wlim)
    m2 = np.abs(allpts[:, 1] - z_med) < 300
    ax[1, 0].scatter(allpts[m2, 3], allpts[m2, 0] / 1000, c=allpts[m2, 2],
                     cmap="RdBu_r", s=4, vmin=-wlim, vmax=wlim,
                     edgecolors="k", linewidths=0.2)
    ax[1, 0].set_title(f"wave Hovmoller (xi,t), zeta={z_med:.0f}mm "
                       f"(stripes slope = c)")
    ax[1, 0].set_xlabel("t (s)"); ax[1, 0].set_ylabel("xi (m)")
    fig.colorbar(im, ax=ax[1, 0])

    # (1,1) 代表点时间序列：原始 PINN + 波成分 + 邻近数据
    ic = np.argmin(np.linalg.norm(allpts[:, :2] - np.array([xi[cx], zeta[cy]]),
                                  axis=1))
    cu, cv = allpts[ic, 0], allpts[ic, 1]
    near = np.linalg.norm(allpts[:, :2] - np.array([cu, cv]), axis=1) < 250
    order = np.argsort(allpts[near, 3])
    ax[1, 1].scatter(allpts[near, 3][order], allpts[near, 2][order], s=6,
                     alpha=0.6, label="data (within 250mm)")
    ix = np.argmin(np.abs(xi - cu))
    iz2 = np.argmin(np.abs(zeta - cv))
    ax[1, 1].plot(ts, eta_g[ix, iz2], color="gray", lw=1, alpha=0.7,
                  label="PINN raw")
    ax[1, 1].plot(ts, eta_w[ix, iz2], "r-", lw=1.5, label="PINN wave comp.")
    ax[1, 1].set_title(f"time series at ({cu:.0f},{cv:.0f})mm")
    ax[1, 1].set_xlabel("t (s)"); ax[1, 1].set_ylabel("eta (mm)")
    ax[1, 1].legend()

    # (1,2) 频谱：原始场 vs 波成分
    sig_raw = eta_g[cx, cy] - eta_g[cx, cy].mean()
    sp_raw = np.abs(np.fft.rfft(sig_raw * np.hanning(len(sig_raw))))
    ax[1, 2].plot(fq, sp_raw / sp_raw.max(), color="gray", label="raw field")
    ax[1, 2].plot(fq, sp / sp.max(), color="r", label="wave comp.")
    ax[1, 2].axvline(F_WAVE_PAPER, color="g", ls="--",
                     label=f"paper {F_WAVE_PAPER}Hz")
    ax[1, 2].set_xlim(0, 3)
    ax[1, 2].set_xlabel("f (Hz)"); ax[1, 2].set_title("spectrum at field center")
    ax[1, 2].legend()
    fig.tight_layout()
    png = os.path.join(OUT, "final_result.png")
    fig.savefig(png, dpi=140)
    print(f"[输出] {png}")

    np.savez_compressed(os.path.join(OUT, "final_field.npz"),
                        xi=xi, zeta=zeta, t=ts, eta=eta_g, eta_wave=eta_w,
                        coverage=cov, rot=rot)
    print(f"[输出] {os.path.join(OUT, 'final_field.npz')}")


if __name__ == "__main__":
    main()
