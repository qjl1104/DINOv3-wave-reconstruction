# wave_modeling/run_real_pinn_honest.py
"""
PINN 去泄漏 + 时间外推诊断（任务 C）。

不修改 run_real_pinn.py，仅 import 复用其函数（含私有辅助 _mad_filter/
_debias/_dedup，只读调用）。三个实验共用同一份训练/评估代码
（train_eval），唯一差异在【传播方向先验的估计集合】与【划分方式】：

- control：完全复刻 run_real_pinn.main() 的原版配方——片段级 80/20 留出
  （prepare_data_split, seed=0），但传播方向/c 先验用 train+test 合并的
  series 估计（measure_direction(series_tr + series_te)，原版 :285 行，
  注释自我辩护为"全局几何先验"）。这是泄漏基线，用于对齐 R²≈0.964。
- (a) 去泄漏片段留出：同一划分（同 seed 同片段），方向/c 先验改为
  【只用 train 片段】估计。与 control 之差即方向先验泄漏的贡献。
- (b) 时间外推：按帧号排序，前 80% 时段的点训练、后 20% 时段预测。
  预处理（PCA 平面、MAD 阈值、归一化 bounds）仍只在 train 时段拟合；
  方向/c 先验只用 train 时段；debias 用各轨迹【train 段】的中位数
  （偏移是时间常数，可由 train 段估计；若用 test 段自身中位数则把
  test 均值信息透给评估，属轻微泄漏——train 段为空的轨迹才回退
  自身中位数并计数报告）。

模型/训练超参与原版完全一致：PINNWaveV2(sigmas=(0.5,0.3,8.0) 旋转系)、
c 固定不参与训练、epochs=2500、lambda_phys=1.0、n_colloc=2048、
train_pinn(seed=0)。注意 (b) 的 train 时段跨度为全长的 0.8 倍，理想
σ_t≈6.4，此处沿用 8.0 以保持与原版可比（B 可学习，偏差 1.25×）。
每个实验在建模前显式 torch.manual_seed(0)，保证 control/(a) 网络初始化
逐参数相同——两者差异只来自方向先验。

R² = 1 − MSE/Var(η_test)（>0 才比均值基线强）；RMSE = sqrt(MSE)（mm）。
测试点 t（及旋转后坐标）可略超出 train 拟合的归一化 bounds，属预期——
对 (b) 而言这正是外推的定义。

用法：
  PYTHONIOENCODING=utf-8 .venv_fs/Scripts/python.exe \
      wave_modeling/run_real_pinn_honest.py --exp all --epochs 2500
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_pinn as rrp              # noqa: E402  （只读复用，勿改原文件）
from pinn_v2 import PINNWaveV2, train_pinn  # noqa: E402

DEVICE = rrp.DEVICE
FPS = rrp.FPS


# ----------------------------------------------------------------------
# 共用训练/评估（与 run_real_pinn.main() 的训练段逐行对应）
# ----------------------------------------------------------------------
def train_eval(pts_tr, pts_te, md, epochs=2500, device=DEVICE, seed=0,
               log_every=500):
    """md = measure_direction 的返回 (c, n_dir) 或 None。
    返回 dict(r2, rmse, ...)。"""
    c_meas, n_dir = (None, None) if md is None else md
    rotated = n_dir is not None
    R = np.eye(2)
    pts_tr = pts_tr.copy()
    pts_te = pts_te.copy()
    if rotated:
        R = np.array([[n_dir[0], n_dir[1]], [-n_dir[1], n_dir[0]]])
        pts_tr[:, :2] = pts_tr[:, :2] @ R.T
        pts_te[:, :2] = pts_te[:, :2] @ R.T

    c_prior = c_meas if c_meas is not None else rrp.C_THEORY
    # bounds 只用训练点（同原版；测试点略超出 [-1,1] 属正常）
    bounds = rrp.compute_bounds(pts_tr)
    xyt = torch.tensor(pts_tr[:, [0, 1, 3]], dtype=torch.float32)
    eta_t = torch.tensor(pts_tr[:, 2:3], dtype=torch.float32)
    xyt_te = torch.tensor(pts_te[:, [0, 1, 3]], dtype=torch.float32)
    eta_te = torch.tensor(pts_te[:, 2:3], dtype=torch.float32)

    sigmas = (0.5, 0.3, 8.0) if rotated else (0.5, 0.5, 8.0)
    torch.manual_seed(seed)   # 固定网络初始化，保证 control/(a) 同初始权重
    model = PINNWaveV2(bounds, c_init=c_prior, learn_c=False, sigmas=sigmas)
    model = train_pinn(model, xyt, eta_t, bounds, epochs=epochs,
                       lambda_phys=1.0, n_colloc=2048, log_every=log_every,
                       device=device, seed=seed)

    model.eval()
    with torch.no_grad():
        pred_te = torch.cat([model.predict(xyt_te[i:i + 4096].to(device))
                             for i in range(0, len(xyt_te), 4096)]).cpu()
    mse = torch.mean((pred_te - eta_te) ** 2).item()
    var = eta_te.var().item()
    corr = float(torch.corrcoef(
        torch.cat([pred_te, eta_te], dim=1).T)[0, 1])
    return {"r2": 1.0 - mse / var, "rmse": float(np.sqrt(mse)), "mse": mse,
            "eta_te_std": float(eta_te.std().item()),
            "pred_mean": float(pred_te.mean()), "pred_std": float(pred_te.std()),
            "corr": corr, "c_prior": float(c_prior),
            "n_dir": None if n_dir is None else tuple(float(x) for x in n_dir),
            "rotated": rotated, "n_tr": len(pts_tr), "n_te": len(pts_te)}


# ----------------------------------------------------------------------
# control / (a)：片段级留出（与原版同一 prepare_data_split, seed=0）
# ----------------------------------------------------------------------
def exp_segment_holdout(pkl, epochs, leak_direction):
    tag = "control(方向先验 train+test 泄漏)" if leak_direction \
        else "(a) 方向先验仅 train"
    print(f"\n{'=' * 72}\n[实验] 片段级留出 seed=0 | {tag}\n{'=' * 72}",
          flush=True)
    d = rrp.prepare_data_split(pkl)
    if leak_direction:
        md = rrp.measure_direction(d["series_tr"] + d["series_te"])  # 原版做法
    else:
        md = rrp.measure_direction(d["series_tr"])                   # 去泄漏
    return train_eval(d["dedup_tr"], d["dedup_te"], md, epochs=epochs)


# ----------------------------------------------------------------------
# (b)：时间外推——前 80% 时段训练，后 20% 时段预测
# ----------------------------------------------------------------------
def prepare_temporal_split(pkl, train_frac=0.8, min_len=20, verbose=True):
    """按帧号切分时段。预处理（PCA/MAD/bounds）只拟合 train 时段；
    debias 用各轨迹 train 段中位数（时间常数偏移，不用 test 段信息）。
    返回 dict(series_tr, dedup_tr, dedup_te, f_split, n_fallback)。"""
    trajs = rrp.load_trajectories(pkl, min_len)
    f_min = min(t[:, 0].min() for t in trajs)
    f_max = max(t[:, 0].max() for t in trajs)
    f_split = f_min + train_frac * (f_max - f_min)

    tr_parts, te_parts = [], []
    for t in trajs:
        m = t[:, 0] <= f_split
        tr_parts.append(t[m])
        te_parts.append(t[~m])
    tr_nonempty = [p for p in tr_parts if len(p) >= 2]

    # PCA 平面只在 train 时段拟合（c0/vt 为基底），MAD 阈值同样 train 拟合。
    # test 时段不单独变换——下方逐轨迹循环用同一基底/MAD 引用重算，以便把
    # 同一轨迹的 train/test 两段配对（debias 需要）。
    series_tr_all, c0, n, vt = rrp.pca_plane_coords(tr_nonempty)
    series_tr_all, eta_med, thr = rrp._mad_filter(series_tr_all)

    # debias：逐轨迹用【train 段】中位数；train 段为空的轨迹回退 test 段
    # 自身中位数（计数报告）。
    e_u, e_v, n_vec = vt[0], vt[1], vt[2]
    series_tr, series_te = [], []
    n_fallback = 0
    for tr_p, te_p in zip(tr_parts, te_parts):
        s_tr = s_te = None
        if len(tr_p) >= 2:
            d = tr_p[:, 1:4] - c0
            s_tr = np.c_[d @ e_u, d @ e_v, d @ n_vec, tr_p[:, 0] / FPS]
            s_tr = s_tr[np.abs(s_tr[:, 2] - eta_med) < thr]
        if len(te_p) >= 2:
            d = te_p[:, 1:4] - c0
            s_te = np.c_[d @ e_u, d @ e_v, d @ n_vec, te_p[:, 0] / FPS]
            s_te = s_te[np.abs(s_te[:, 2] - eta_med) < thr]
        if s_tr is None and s_te is None:
            continue
        if s_tr is not None and len(s_tr):
            med = np.median(s_tr[:, 2])
        else:                                   # train 段缺失 → 回退
            med = np.median(s_te[:, 2])
            n_fallback += 1
        if s_tr is not None and len(s_tr):
            s_tr[:, 2] -= med
            series_tr.append(s_tr)
        if s_te is not None and len(s_te):
            s_te[:, 2] -= med
            series_te.append(s_te)

    dedup_tr, dedup_te = rrp._dedup(series_tr), rrp._dedup(series_te)
    if verbose:
        print(f"[时间划分] 帧 {f_min:.0f}–{f_max:.0f}，切于 {f_split:.0f} "
              f"(t={f_split / FPS:.2f}s，前 {train_frac:.0%} 时段) | "
              f"train {len(series_tr)} 段 {len(dedup_tr)} 点 / "
              f"test {len(series_te)} 段 {len(dedup_te)} 点 | "
              f"train η std = {dedup_tr[:, 2].std():.2f} mm, "
              f"test η std = {dedup_te[:, 2].std():.2f} mm | "
              f"debias 回退自身中位数的轨迹 {n_fallback} 条")
    return {"series_tr": series_tr, "dedup_tr": dedup_tr, "dedup_te": dedup_te,
            "f_split": f_split, "n_fallback": n_fallback}


def exp_temporal_extrapolation(pkl, epochs, train_frac=0.8):
    print(f"\n{'=' * 72}\n[实验] (b) 时间外推：前 {train_frac:.0%} 时段 → "
          f"后 {1 - train_frac:.0%} 时段 | 方向先验仅 train\n{'=' * 72}",
          flush=True)
    d = prepare_temporal_split(pkl, train_frac=train_frac)
    md = rrp.measure_direction(d["series_tr"])   # 仅 train 时段
    res = train_eval(d["dedup_tr"], d["dedup_te"], md, epochs=epochs)
    res["f_split"] = d["f_split"]
    res["n_fallback"] = d["n_fallback"]
    # 外推失败形态诊断：pred std≈η std 且 corr≈0 → 相位去相关（幅度对、
    # 相位错）；pred std 远大于 η std → 幅度发散
    print(f"[(b) 预测诊断] pred mean={res['pred_mean']:+.2f} mm, "
          f"pred std={res['pred_std']:.2f} mm（真值 std="
          f"{res['eta_te_std']:.2f}）, corr(pred,true)={res['corr']:+.3f}",
          flush=True)
    return res


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="PINN 去泄漏 + 时间外推诊断（不改 run_real_pinn.py）")
    ap.add_argument("pkl", nargs="?", default=rrp.PKL)
    ap.add_argument("--exp", default="all",
                    choices=["all", "control", "a", "b"],
                    help="control=原版泄漏基线；a=方向先验仅train的片段留出；"
                         "b=时间外推")
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--train-frac", type=float, default=0.8)
    args = ap.parse_args()
    print(f"[输入] {args.pkl} | device={DEVICE} | epochs={args.epochs}",
          flush=True)

    results = {}
    t0 = time.time()
    if args.exp in ("all", "control"):
        results["control"] = exp_segment_holdout(args.pkl, args.epochs,
                                                 leak_direction=True)
        print(f"[control] R²={results['control']['r2']:.4f} "
              f"RMSE={results['control']['rmse']:.2f}mm "
              f"({time.time() - t0:.0f}s)", flush=True)
    if args.exp in ("all", "a"):
        t1 = time.time()
        results["a"] = exp_segment_holdout(args.pkl, args.epochs,
                                           leak_direction=False)
        print(f"[(a)] R²={results['a']['r2']:.4f} "
              f"RMSE={results['a']['rmse']:.2f}mm "
              f"({time.time() - t1:.0f}s)", flush=True)
    if args.exp in ("all", "b"):
        t2 = time.time()
        results["b"] = exp_temporal_extrapolation(args.pkl, args.epochs,
                                                  args.train_frac)
        print(f"[(b)] R²={results['b']['r2']:.4f} "
              f"RMSE={results['b']['rmse']:.2f}mm "
              f"({time.time() - t2:.0f}s)", flush=True)

    # ---------------- 对比表 ----------------
    print(f"\n{'=' * 92}\n[对比表] PINN 留出/外推评估（epochs={args.epochs}, "
          f"train_pinn seed=0）\n{'=' * 92}")
    head = (f"{'实验':<44} {'R²':>8} {'RMSE(mm)':>9} {'η_te std':>9} "
            f"{'c先验':>7} {'方向':>18} {'n_tr':>8} {'n_te':>8}")
    print(head)
    print("-" * len(head))
    labels = {
        "control": "原版(方向先验 train+test, 泄漏)",
        "a": "(a) 片段留出, 方向先验仅 train",
        "b": f"(b) 时间外推 前{args.train_frac:.0%}→后{1 - args.train_frac:.0%}, 先验仅 train",
    }
    for k in ("control", "a", "b"):
        if k not in results:
            continue
        r = results[k]
        nd = "None" if r["n_dir"] is None else \
            f"({r['n_dir'][0]:.3f},{r['n_dir'][1]:.3f})"
        print(f"{labels[k]:<44} {r['r2']:>8.4f} {r['rmse']:>9.2f} "
              f"{r['eta_te_std']:>9.2f} {r['c_prior']:>7.0f} {nd:>18} "
              f"{r['n_tr']:>8} {r['n_te']:>8}")
    if "control" in results and "a" in results:
        d_r2 = results["control"]["r2"] - results["a"]["r2"]
        d_rmse = results["a"]["rmse"] - results["control"]["rmse"]
        print(f"\n[泄漏量] control − (a): ΔR² = {d_r2:+.4f}, "
              f"ΔRMSE = {d_rmse:+.2f} mm（方向先验泄漏的贡献）")
    if "b" in results:
        print(f"[外推] (b) R²={results['b']['r2']:.4f}：>0 才优于均值基线；"
              f"与 (a) 的差距 = 插值技能 → 外推技能的落差")
    print(f"[总耗时] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
