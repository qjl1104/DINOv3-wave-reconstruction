# wave_modeling/verify_results.py
"""
结果核验包：生成一张"一眼核验"图 + 打印核验清单。
内容：
1. 最长 3 条轨迹的 η(t) 时间序列（debias 后）——应肉眼可见 0.79Hz 正弦；
2. 各自的 FFT 谱——主峰应对齐论文 0.79Hz；
3. 重建场波成分在该轨迹位置的重合度（PINN vs 数据）；
4. 打印关键指标汇总（R²、c、主频、波长、振幅）。
用法：.venv_fs/Scripts/python.exe wave_modeling/verify_results.py
输出：wave_modeling/real_run/verify_results.png
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_v2 import PINNWaveV2  # noqa: E402
from run_real_pinn import C_THEORY, FPS, OUT, prepare_data  # noqa: E402

F_WAVE = 0.79
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _regrid_uniform(t_sec, *ys, max_interp_gap=4):
    """把同一时间基上的多条序列按真实帧号重采样到均匀帧网格（缺帧不能被
    FFT 当作均匀采样，否则频率最多偏 ~6%）：≤max_interp_gap 帧的短空洞
    （≤0.08s ≈ 1/10 周期）线性插值；更长的空洞零填充——序列已 debias/
    带通居中，0 ≈ 均值电平，且窄带单频信号在零填充下相位近似无偏，而
    线性插值会在 >1/4 周期的空洞上注入错误相位（实测最长空洞 24 帧
    = 0.48s ≈ 0.37 周期）。无空洞时插值节点即原采样点，等价恒等。
    返回 (t_uniform, *ys_uniform)。"""
    fr = np.round(t_sec * FPS).astype(int)
    order = np.argsort(fr)
    fr = fr[order]
    _, first = np.unique(fr, return_index=True)
    fr = fr[first]
    grid = np.arange(fr[0], fr[-1] + 1)
    out = [np.interp(grid, fr, y[order][first]) for y in ys]
    long = np.flatnonzero(np.diff(fr) - 1 > max_interp_gap)
    for i in long:                       # 长空洞段改回 0（均值电平）
        sl = slice(fr[i] + 1 - fr[0], fr[i + 1] - fr[0])
        for o in out:
            o[sl] = 0.0
    return (grid / FPS,) + tuple(out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pkl = os.path.join(OUT, "..", "..", "data/trajectories/trajectories_3d_v2_dino.pkl")
    series, allpts, _ = prepare_data(pkl)
    ck = torch.load(os.path.join(OUT, "pinn_real.pt"), map_location="cpu",
                    weights_only=False)
    rot = ck.get("rot", np.eye(2))
    model = PINNWaveV2(ck["bounds"], c_init=C_THEORY, learn_c=False)
    model.load_state_dict(ck["model"])
    model.eval().to(DEVICE)

    # 找最长 3 条片段
    order = np.argsort([len(s) for s in series])[::-1][:3]
    fig, axes = plt.subplots(3, 3, figsize=(18, 9))
    print("=== 核验清单 ===")

    def bandpass(e):
        E = np.fft.rfft(e - e.mean())
        fq = np.fft.rfftfreq(len(e), 1 / FPS)
        E[(fq < 0.5) | (fq > 1.2)] = 0.0
        return np.fft.irfft(E, len(e))

    for row, si in enumerate(order):
        s = series[si]
        s_rot = s.copy()
        s_rot[:, :2] = s_rot[:, :2] @ rot.T
        t_, eta_ = s_rot[:, 3], s_rot[:, 2]
        # PINN 先在相同（缺帧）采样点上预测，再与数据一起插回均匀帧时间基
        q = torch.tensor(s_rot[:, [0, 1, 3]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred = model.predict(q).cpu().numpy().ravel()
        # 帧号含缺帧：带通/FFT 前把数据与 PINN 序列重采样到均匀帧网格
        t_u, eta_u, pred_u = _regrid_uniform(t_, eta_, pred)
        eta_w = bandpass(eta_u)
        pred_w = bandpass(pred_u)
        # 左：原始 η(t)（可见漂移 + 波动）
        ax = axes[row, 0]
        ax.plot(t_, eta_, ".-", ms=3, lw=0.8)
        ax.set_title(f"fragment #{si}: raw η(t), {len(s)} pts "
                     f"({t_.min():.1f}–{t_.max():.1f}s)")
        ax.set_xlabel("t (s)"); ax.set_ylabel("η (mm)")
        # 中：带通后 η(t) vs PINN（纯波成分重合度）
        ax = axes[row, 1]
        ax.plot(t_u, eta_w, ".", ms=3, label="data (bandpassed)")
        ax.plot(t_u, pred_w, "-", lw=1.2, label="PINN (bandpassed)", alpha=0.85)
        ss_res = np.sum((eta_w - pred_w) ** 2)
        ss_tot = np.sum((eta_w - eta_w.mean()) ** 2)
        ax.set_title(f"wave comp. (0.5-1.2Hz): R²={1 - ss_res / max(ss_tot, 1e-9):.2f}")
        ax.set_xlabel("t (s)")
        ax.legend(markerscale=2)
        # 右：FFT（原始 vs 带通）
        e = eta_u - eta_u.mean()
        w = np.hanning(len(e))
        sp = np.abs(np.fft.rfft(e * w))
        spw = np.abs(np.fft.rfft(eta_w * w))
        fq = np.fft.rfftfreq(len(e), 1 / FPS)
        ax = axes[row, 2]
        ax.plot(fq, sp / sp.max(), color="gray", label="raw")
        ax.plot(fq, spw / spw.max(), "r", label="bandpassed")
        ax.axvline(F_WAVE, color="g", ls="--")
        ax.set_xlim(0, 3)
        kw = np.argmax(spw[(fq > 0.4) & (fq < 3)]) + np.argmax(fq > 0.4)
        amp = 2 * spw[kw] / w.sum()   # 窗增益修正：幅值 = 2|X|/Σw（Hann 即 4|X|/N）
        ax.set_title(f"FFT: raw dom {fq[np.argmax(sp[1:]) + 1]:.2f}Hz → "
                     f"wave {fq[kw]:.2f}Hz, amp {amp:.1f}mm")
        ax.set_xlabel("f (Hz)")
        ax.legend()
        print(f"片段 #{si}（{len(s)} 帧）：原始谱主峰 {fq[np.argmax(sp[1:]) + 1]:.2f} Hz"
              f"（低频漂移，师兄论文亦记载此现象）；带通后主峰 {fq[kw]:.3f} Hz、"
              f"振幅 {amp:.1f} mm")
    fig.tight_layout()
    png = os.path.join(OUT, "verify_results.png")
    fig.savefig(png, dpi=140)
    print(f"[输出] {png}")
    print("复算全量指标：.venv_fs/Scripts/python.exe wave_modeling/eval_tracks.py "
          "data/trajectories/trajectories_3d_v2_dino.pkl 核验")
    print("一键复现全链：.venv_fs/Scripts/python.exe run_full_pipeline.py")


if __name__ == "__main__":
    main()
