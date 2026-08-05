# particle_processing/thin_density_sweep.py
"""
粒子密度抽稀扫描驱动（开题"极端稀疏工况性能边界"实验）。

对每一档 (p, seed) 串起完整下游链路：
  thin（已抽稀检测，thin_detections.py）
  → 03b_tracker_appearance.py --no-foam（2D 轨迹）
  → clean_tracks_jumpcut.py（跳切清洗）
  → DINOv3/compute_desc_tracks.py（描述子，双侧）
  → rematch_dino_v2.py --hung-only（3D 匹配）
  → wave_modeling/eval_tracks.py（η std / 主峰率 / c）
  → wave_modeling/eval_c_block_bootstrap.py（c 的诚实 CI，约 1-2 min/档；
    低密度档有效对不足会失败，捕获并记为失效——本身就是性能边界数据）

幂等：各阶段产物存在且比输入新则跳过；CSV 已有该 (p,seed) 的 ok 行且 3D 产物
仍在则整档跳过。每档结束（含失败）立即向 CSV 追加一行，中途超时/中断不丢进度，
重跑本脚本即可续跑。

产物路径（全部为新路径，绝不触碰 canonical）：
  data/detections/thin/detections_{side}_v3_thin{pct}_s{seed}.pkl(+_meta)
  data/trajectories/thin/trajectories_2d_{side}_v3nf_thin{pct}_s{seed}.pkl(+_jumpcut)
  DINOv3/desc_thin{pct}_s{seed}_{side}.pkl
  data/trajectories/thin/trajectories_3d_thin{pct}_s{seed}.pkl
  wave_modeling/real_run/thin_density_sweep_results.csv（逐档追加）
  wave_modeling/real_run/thin_density_sweep_stages.log（各阶段完整 stdout）

用法：
  ../.venv_fs/Scripts/python.exe thin_density_sweep.py                 # 全部档位
  ../.venv_fs/Scripts/python.exe thin_density_sweep.py --p 0.5 --seeds 1
"""

import argparse
import csv
import os
import pickle
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

DET_DIR = os.path.join(ROOT, "data", "detections")
THIN_DET_DIR = os.path.join(DET_DIR, "thin")
THIN_TRAJ_DIR = os.path.join(ROOT, "data", "trajectories", "thin")
DINO_DIR = os.path.join(ROOT, "DINOv3")
CSV_PATH = os.path.join(ROOT, "wave_modeling", "real_run", "thin_density_sweep_results.csv")
LOG_PATH = os.path.join(ROOT, "wave_modeling", "real_run", "thin_density_sweep_stages.log")

# (p, seeds)；p=1.0 用 canonical 既有结果，不在此驱动内
ALL_COMBOS = [(0.5, [1]), (0.25, [1]), (0.1, [1]),
              (0.05, [1, 2, 3]), (0.01, [1, 2, 3])]

CSV_FIELDS = ["p", "seed", "n_det_left", "n_det_right",
              "n_traj2d_left", "n_traj2d_right",
              "n_3d_seg", "n_3d_pts", "eta_std_mm", "peak_rate_pct", "n_ge100",
              "c_mm_s", "ci_lo", "ci_hi", "status", "elapsed_s", "note"]


def pct(p):
    return str(int(round(p * 100)))


def paths(p, seed):
    tag = f"thin{pct(p)}_s{seed}"
    det = {s: os.path.join(THIN_DET_DIR, f"detections_{s}_v3_{tag}.pkl")
           for s in ("left", "right")}
    t2d = {s: os.path.join(THIN_TRAJ_DIR, f"trajectories_2d_{s}_v3nf_{tag}.pkl")
           for s in ("left", "right")}
    t2djc = {s: os.path.join(THIN_TRAJ_DIR,
                             f"trajectories_2d_{s}_v3nf_{tag}_jumpcut.pkl")
             for s in ("left", "right")}
    desc = {s: os.path.join(DINO_DIR, f"desc_{tag}_{s}.pkl")
            for s in ("left", "right")}
    t3d = os.path.join(THIN_TRAJ_DIR, f"trajectories_3d_{tag}.pkl")
    return det, t2d, t2djc, desc, t3d


def newest_mtime(ps):
    return max(os.path.getmtime(x) for x in ps if os.path.exists(x)) if any(
        os.path.exists(x) for x in ps) else 0


def stale(outputs, inputs):
    if not all(os.path.exists(o) for o in outputs):
        return True
    return min(os.path.getmtime(o) for o in outputs) < newest_mtime(inputs)


def log_full(text):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_stage(stage, cmd, cwd):
    """跑一个阶段，返回 (ok, stdout)。完整输出进 stages.log。"""
    print(f"\n[{stage}] {' '.join(os.path.basename(c) if c.endswith('.py') else c for c in cmd)}",
          flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=ENV,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    out = r.stdout or ""
    dt = time.time() - t0
    log_full(f"\n{'=' * 70}\n[{stage}] {' '.join(cmd)} (cwd={cwd}) "
             f"rc={r.returncode} {dt:.0f}s\n{'=' * 70}\n{out}")
    tail = "\n".join(out.rstrip().splitlines()[-6:])
    print(f"[{stage}] rc={r.returncode}（{dt:.0f}s）\n{tail}", flush=True)
    return r.returncode == 0, out


def count_det_points(pkl_path):
    with open(pkl_path, "rb") as f:
        dets = pickle.load(f)
    return sum(len(d) for d in dets)


def count_tracks(pkl_path):
    """2D 轨迹 pkl 是 __main__.Track 系列，打桩后 unpickle 计数（缓存命中时用）。"""
    sys.path.insert(0, HERE)
    import rematch_rectified as rr  # noqa: E402
    import __main__  # noqa: E402
    for _n in ["Track", "UltraTrack", "WaveParticleTrack", "StrictTrack",
               "SimpleKalmanFilter", "ImprovedKalmanFilter",
               "ExtendedKalmanFilter", "OptimizedExtendedKalmanFilter",
               "UltraOptimizedKalmanFilter", "WaveParticleKalmanFilter",
               "StrictWaveKalmanFilter"]:
        setattr(__main__, _n, getattr(rr, _n))
    __main__.RobustKalmanFilter = type(
        "RobustKalmanFilter", (rr.SimpleKalmanFilter,), {})
    with open(pkl_path, "rb") as f:
        return len(pickle.load(f))


def parse_traj2d(stdout):
    n = {}
    for m in re.finditer(r"^(left|right): 轨迹 (\d+) 条", stdout, re.M):
        n[m.group(1)] = int(m.group(2))
    return n


def parse_eval(stdout):
    """解析 eval_tracks.py 输出，返回 dict（缺项为 None）。"""
    r = {"n_3d_seg": None, "n_3d_pts": None, "eta_std_mm": None,
         "peak_rate_pct": None, "n_ge100": None, "c_mm_s": None}
    m = re.search(r"\[量\] 片段 (\d+) 条", stdout)
    if m:
        r["n_3d_seg"] = int(m.group(1))
    m = re.search(r"总点 (\d+)", stdout)
    if m:
        r["n_3d_pts"] = int(m.group(1))
    m = re.search(r"η std = ([\d.]+) mm", stdout)
    if m:
        r["eta_std_mm"] = float(m.group(1))
    m = re.search(r"≥100帧片段 (\d+) 条 \| 主峰 0\.79±0\.1Hz 比例 (\d+)%", stdout)
    if m:
        r["n_ge100"] = int(m.group(1))
        r["peak_rate_pct"] = int(m.group(2))
    m = re.search(r"\[相\] c = (\d+) mm/s", stdout)
    if m:
        r["c_mm_s"] = int(m.group(1))
    return r


def parse_bootstrap(stdout):
    """解析片段级 cluster bootstrap 的 95% CI；失败返回 (None, None)。"""
    m = re.search(r"片段级 cluster bootstrap\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
                  r"\s*([\d.]+)\s*\|\s*([\d.]+)", stdout)
    if m:
        return float(m.group(3)), float(m.group(4))
    return None, None


def csv_done():
    """CSV 中已 ok 完成的 (p, seed) 集合。"""
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add((float(row["p"]), int(row["seed"])))
    return done


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k, ""))
                    for k in CSV_FIELDS})
    print(f"[CSV] 已追加 p={row['p']} s={row['seed']} status={row['status']}",
          flush=True)


def run_combo(p, seed):
    tag = f"thin{pct(p)}_s{seed}"
    label = f"抽稀 p={p} seed={seed}"
    det, t2d, t2djc, desc, t3d = paths(p, seed)
    canon_det = {s: os.path.join(DET_DIR, f"detections_{s}_v3.pkl")
                 for s in ("left", "right")}
    row = {"p": p, "seed": seed, "status": "ok", "note": ""}
    t_combo = time.time()

    def fail(stage, note=""):
        row["status"] = f"failed@{stage}"
        row["note"] = note.replace("\n", " ")[:200]
        row["elapsed_s"] = round(time.time() - t_combo)
        append_row(row)
        return False

    # [1] 抽稀（正常已被预生成；缺失则补）
    if stale(list(det.values()), list(canon_det.values())):
        ok, _ = run_stage(f"{tag}/thin",
                          [PY, os.path.join(HERE, "thin_detections.py"),
                           str(p), str(seed)], ROOT)
        if not ok:
            return fail("thin")
    else:
        print(f"[{tag}/thin] 已缓存且未过期，跳过", flush=True)
    for s in ("left", "right"):
        row[f"n_det_{s}"] = count_det_points(det[s])

    # [2] 跟踪（--no-foam，水沫在跟踪侧剔除，与生产链一致）
    if stale(list(t2d.values()), list(det.values())):
        ok, out = run_stage(f"{tag}/track",
                            [PY, os.path.join(HERE, "03b_tracker_appearance.py"),
                             det["left"], det["right"],
                             t2d["left"], t2d["right"], "--no-foam"], ROOT)
        if not ok:
            return fail("track")
        for s, n in parse_traj2d(out).items():
            row[f"n_traj2d_{s}"] = n
    else:
        print(f"[{tag}/track] 已缓存且未过期，跳过", flush=True)
        for s in ("left", "right"):
            row[f"n_traj2d_{s}"] = count_tracks(t2d[s])

    # [3] 跳切清洗
    if stale(list(t2djc.values()), list(t2d.values())):
        ok, _ = run_stage(f"{tag}/jumpcut",
                          [PY, os.path.join(HERE, "clean_tracks_jumpcut.py"),
                           "--src", t2d["left"], t2d["right"],
                           "--dst", t2djc["left"], t2djc["right"]], ROOT)
        if not ok:
            return fail("jumpcut")
    else:
        print(f"[{tag}/jumpcut] 已缓存且未过期，跳过", flush=True)

    # [4] DINO 描述子（双侧，DINOv3 目录下跑）
    for s in ("left", "right"):
        if stale([desc[s]], [t2djc[s]]):
            ok, _ = run_stage(f"{tag}/desc-{s}",
                              [PY, "compute_desc_tracks.py", s,
                               t2djc[s], desc[s]], DINO_DIR)
            if not ok:
                return fail(f"desc-{s}")
        else:
            print(f"[{tag}/desc-{s}] 已缓存且未过期，跳过", flush=True)

    # [5] 匈牙利一对一匹配（particle_processing 目录下跑）
    if stale([t3d], list(t2djc.values()) + list(desc.values())):
        ok, _ = run_stage(f"{tag}/match",
                          [PY, "rematch_dino_v2.py",
                           t2djc["left"], t2djc["right"],
                           desc["left"], desc["right"], t3d, "--hung-only"], HERE)
        if not ok:
            return fail("match")
    else:
        print(f"[{tag}/match] 已缓存且未过期，跳过", flush=True)

    # [6] 评估（每档必跑，是裁判）
    ok, out = run_stage(f"{tag}/eval",
                        [PY, os.path.join(ROOT, "wave_modeling", "eval_tracks.py"),
                         t3d, label], ROOT)
    if not ok:
        return fail("eval", "eval_tracks 退出非零（可能 3D 片段为空）")
    row.update(parse_eval(out))
    if "MAD 剔除后无剩余片段" in out:
        row["note"] = (row["note"] + " MAD后无片段").strip()
    if "有效对不足" in out:
        row["note"] = (row["note"] + " c有效对不足").strip()

    # [7] c 的诚实 CI（低密度档可能失效，捕获不中止）。
    # 注意：eval_c_block_bootstrap.py 固定把诊断图写到
    # real_run/eval_c_block_bootstrap.png（canonical 审计产物）——先备份，
    # 跑完把新图改名为档位专属，再恢复备份，避免覆盖既有文件。
    bs_png = os.path.join(ROOT, "wave_modeling", "real_run",
                          "eval_c_block_bootstrap.png")
    bs_bak = bs_png + ".bak_thin_sweep"
    had_png = os.path.exists(bs_png)
    if had_png:
        shutil.copy2(bs_png, bs_bak)
    ok, out = run_stage(f"{tag}/c-ci",
                        [PY, os.path.join(ROOT, "wave_modeling",
                                          "eval_c_block_bootstrap.py"), t3d], ROOT)
    if os.path.exists(bs_png):
        os.replace(bs_png, bs_png.replace(".png", f"_{tag}.png"))
    if had_png:
        shutil.move(bs_bak, bs_png)
    if ok:
        lo, hi = parse_bootstrap(out)
        row["ci_lo"], row["ci_hi"] = lo, hi
        if lo is None:
            row["note"] = (row["note"] + " CI解析失败").strip()
    else:
        row["note"] = (row["note"] + " bootstrap失效").strip()

    row["elapsed_s"] = round(time.time() - t_combo)
    append_row(row)
    print(f"[{tag}] 档位完成，用时 {row['elapsed_s']}s", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=None, help="只跑该保留概率档")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="只跑这些 seed")
    args = ap.parse_args()

    os.makedirs(THIN_TRAJ_DIR, exist_ok=True)
    done = csv_done()
    combos = [(p, s) for p, seeds in ALL_COMBOS for s in seeds
              if (args.p is None or abs(p - args.p) < 1e-9)
              and (args.seeds is None or s in args.seeds)]
    print(f"计划档位：{[(p, s) for p, s in combos]}", flush=True)
    t_all = time.time()
    for p, seed in combos:
        det, t2d, t2djc, desc, t3d = paths(p, seed)
        if (p, seed) in done and os.path.exists(t3d):
            print(f"\n[p={p} s={seed}] CSV 已有 ok 行且 3D 产物在，整档跳过",
                  flush=True)
            continue
        try:
            run_combo(p, seed)
        except Exception as e:  # 单档异常不中止后续档
            print(f"[p={p} s={seed}] 未捕获异常：{e!r}，继续下一档", flush=True)
            append_row({"p": p, "seed": seed, "status": "failed@exception",
                        "note": repr(e)[:200], "elapsed_s": 0})
    print(f"\n全部计划档位结束，总用时 {(time.time() - t_all) / 60:.1f} 分钟",
          flush=True)


if __name__ == "__main__":
    main()
