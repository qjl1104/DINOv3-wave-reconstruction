# particle_processing/thin_tracks_density_sweep.py
"""
轨迹级抽稀扫描（物理稀疏模型："少撒粒子"=整标记移除，不打断跟踪，只考验匹配）。

与 thin_density_sweep.py（可见性模型：逐帧 Bernoulli 抽稀检测点）互补：
本脚本从 canonical 2D 轨迹 + DINO 描述子出发，按保留概率 p 随机丢【整条】
轨迹，重跑匹配与评估。无跟踪/DINO 重算，单档 ~1-5 分钟。

口径：左右两侧独立抽稀（同 seed、独立随机流），同一物理粒子须两侧都存活才
能配上 → 有效物理密度 q = p²。汇总以 q 为横轴。

正确性前提（已验证 2026-08-05）：desc pkl 是与轨迹列表 1:1 对齐的
list[dict{frame: (768,) fp32}]，且每条 desc 的帧键是对应轨迹 points 帧键的
子集（双侧全量核验 0 错位）→ 同一掩码同时过滤两列表即保持一致。
p=1.0（不抽）必须复现 canonical：975 段 / 292039 点 / c=1992，否则
status=validation_failed 并整体停止（说明 desc 过滤错位）。

产物（全部新路径，canonical 只读）：
  data/trajectories/thin_traj/trajectories_2d_{side}_v3nf_thintraj{pct}_s{seed}.pkl
  DINOv3/desc_thintraj{pct}_s{seed}_{side}.pkl
  data/trajectories/thin_traj/trajectories_3d_thintraj{pct}_s{seed}.pkl
  wave_modeling/real_run/thin_traj_density_sweep_results.csv（逐档追加）
  wave_modeling/real_run/thin_traj_density_sweep_stages.log
  wave_modeling/real_run/eval_c_block_bootstrap_thintraj{pct}_s{seed}.png
  （bootstrap 脚本固定覆盖 real_run/eval_c_block_bootstrap.png：
    备份→改名→恢复护栏，与上一轮相同）

幂等：CSV 已有该 (p,seed) 的 ok 行且 3D 产物在 → 整档跳过；抽稀子集已存在
→ 跳过重写。中断后重跑本脚本即续跑。

用法：
  ../.venv_fs/Scripts/python.exe thin_tracks_density_sweep.py              # 全部档位
  ../.venv_fs/Scripts/python.exe thin_tracks_density_sweep.py --p 0.5 --seeds 1
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

CANON_TRAJ = {s: os.path.join(ROOT, "data", "trajectories",
                              f"trajectories_2d_{s}_v3nf_jumpcut.pkl")
              for s in ("left", "right")}
CANON_DESC = {s: os.path.join(ROOT, "DINOv3", f"desc_v3nftracks_{s}.pkl")
              for s in ("left", "right")}
THIN_DIR = os.path.join(ROOT, "data", "trajectories", "thin_traj")
DINO_DIR = os.path.join(ROOT, "DINOv3")
CSV_PATH = os.path.join(ROOT, "wave_modeling", "real_run",
                        "thin_traj_density_sweep_results.csv")
LOG_PATH = os.path.join(ROOT, "wave_modeling", "real_run",
                        "thin_traj_density_sweep_stages.log")

# p 从大到小；p=1.0 为一致性校验档（直接用 canonical 输入，不写子集）
ALL_COMBOS = [(1.0, [1]), (0.5, [1]), (0.25, [1]), (0.1, [1]),
              (0.05, [1, 2, 3]), (0.01, [1, 2, 3])]

# canonical 基线（eval 口径），p=1.0 校验锚
CANON_SEG, CANON_PTS, CANON_C = 975, 292039, 1992

CSV_FIELDS = ["p", "q", "seed", "n_traj_left", "n_traj_right",
              "n_3d_seg", "n_3d_pts", "eta_std_mm", "peak_rate_pct", "n_ge100",
              "c_mm_s", "ci_lo", "ci_hi", "status", "elapsed_s", "note"]

_cache = {}  # side -> (trajs, desc)，canonical 只加载一次


def pct(p):
    return str(int(round(p * 100)))


def tag_of(p, seed):
    return f"thintraj{pct(p)}_s{seed}"


def paths(p, seed):
    tag = tag_of(p, seed)
    traj = {s: os.path.join(THIN_DIR, f"trajectories_2d_{s}_v3nf_{tag}.pkl")
            for s in ("left", "right")}
    desc = {s: os.path.join(DINO_DIR, f"desc_{tag}_{s}.pkl")
            for s in ("left", "right")}
    t3d = os.path.join(THIN_DIR, f"trajectories_3d_{tag}.pkl")
    return traj, desc, t3d


def load_canonical():
    """加载 canonical 轨迹 + 描述子（类桩模式，与生产链各脚本一致）。"""
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
    for s in ("left", "right"):
        if s in _cache:
            continue
        with open(CANON_TRAJ[s], "rb") as f:
            trajs = pickle.load(f)
        with open(CANON_DESC[s], "rb") as f:
            desc = pickle.load(f)
        assert len(trajs) == len(desc), f"{s}: 轨迹/desc 条数不一致"
        _cache[s] = (trajs, desc)
        print(f"[加载] {s}: {len(trajs)} 条轨迹 + desc", flush=True)


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


def parse_eval(stdout):
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
    m = re.search(r"片段级 cluster bootstrap\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
                  r"\s*([\d.]+)\s*\|\s*([\d.]+)", stdout)
    if m:
        return float(m.group(3)), float(m.group(4))
    return None, None


def csv_done():
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
    tag = tag_of(p, seed)
    label = f"轨迹抽稀 p={p} seed={seed}"
    traj, desc, t3d = paths(p, seed)
    row = {"p": p, "q": round(p * p, 6), "seed": seed,
           "status": "ok", "note": ""}
    t_combo = time.time()

    def fail(stage, note=""):
        row["status"] = f"failed@{stage}"
        row["note"] = note.replace("\n", " ")[:200]
        row["elapsed_s"] = round(time.time() - t_combo)
        append_row(row)
        return False

    # [1] 轨迹级抽稀子集（p=1.0 直接用 canonical 输入，不落盘）
    if p >= 1.0:
        in_traj, in_desc = dict(CANON_TRAJ), dict(CANON_DESC)
        row["n_traj_left"] = len(_cache["left"][0])
        row["n_traj_right"] = len(_cache["right"][0])
    else:
        in_traj, in_desc = traj, desc
        for si, s in enumerate(("left", "right")):
            if stale([traj[s], desc[s]], [CANON_TRAJ[s], CANON_DESC[s]]):
                trajs, dscs = _cache[s]
                # 同 seed、两侧独立随机流：逐轨迹 Bernoulli(p)
                rng = np.random.default_rng([seed, si])
                mask = rng.random(len(trajs)) < p
                idx = np.flatnonzero(mask)
                os.makedirs(THIN_DIR, exist_ok=True)
                with open(traj[s], "wb") as f:
                    pickle.dump([trajs[i] for i in idx], f)
                with open(desc[s], "wb") as f:
                    pickle.dump([dscs[i] for i in idx], f)
                print(f"[{tag}/subset] {s}: {len(trajs)} → {len(idx)} 条"
                      f"（{len(idx) / len(trajs):.3f}）", flush=True)
            else:
                print(f"[{tag}/subset-{s}] 已缓存且未过期，跳过", flush=True)
            # 轨迹数：desc 列表与轨迹列表 1:1，读 desc 长度即可（缓存命中亦适用）
            with open(desc[s], "rb") as f:
                row[f"n_traj_{s}"] = len(pickle.load(f))

    # [2] 匈牙利一对一匹配（particle_processing 目录下跑）
    if stale([t3d], list(in_traj.values()) + list(in_desc.values())):
        ok, _ = run_stage(f"{tag}/match",
                          [PY, "rematch_dino_v2.py",
                           in_traj["left"], in_traj["right"],
                           in_desc["left"], in_desc["right"],
                           t3d, "--hung-only"], HERE)
        if not ok:
            return fail("match")
    else:
        print(f"[{tag}/match] 已缓存且未过期，跳过", flush=True)

    # [3] 评估（每档必跑，是裁判）
    ok, out = run_stage(f"{tag}/eval",
                        [PY, os.path.join(ROOT, "wave_modeling",
                                          "eval_tracks.py"), t3d, label], ROOT)
    if not ok:
        return fail("eval", "eval_tracks 退出非零（可能 3D 片段为空）")
    row.update(parse_eval(out))
    if "MAD 剔除后无剩余片段" in out:
        row["note"] = (row["note"] + " MAD后无片段").strip()
    if "有效对不足" in out:
        row["note"] = (row["note"] + " c有效对不足").strip()

    # [4] c 的诚实 CI（备份→改名→恢复护栏，保护 canonical 审计图）
    bs_png = os.path.join(ROOT, "wave_modeling", "real_run",
                          "eval_c_block_bootstrap.png")
    bs_bak = bs_png + ".bak_thin_traj_sweep"
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

    # p=1.0 一致性校验：必须逐位复现 canonical，否则停止（desc 对齐错位）
    if p >= 1.0:
        good = (row["n_3d_seg"] == CANON_SEG and row["n_3d_pts"] == CANON_PTS
                and row["c_mm_s"] == CANON_C)
        if not good:
            append_row({"p": p, "q": 1.0, "seed": seed,
                        "status": "validation_failed",
                        "note": f"期望 {CANON_SEG}/{CANON_PTS}/{CANON_C}，"
                                f"实得 {row['n_3d_seg']}/{row['n_3d_pts']}/{row['c_mm_s']}",
                        "elapsed_s": 0})
            sys.exit("[校验失败] p=1.0 未复现 canonical，停止全部档位，"
                     "请检查 desc 对齐")
        print("[校验通过] p=1.0 复现 canonical："
              f"{CANON_SEG} 段 / {CANON_PTS} 点 / c={CANON_C}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(THIN_DIR, exist_ok=True)
    load_canonical()
    done = csv_done()
    combos = [(p, s) for p, seeds in ALL_COMBOS for s in seeds
              if (args.p is None or abs(p - args.p) < 1e-9)
              and (args.seeds is None or s in args.seeds)]
    print(f"计划档位：{[(p, s) for p, s in combos]}", flush=True)
    t_all = time.time()
    for p, seed in combos:
        _, _, t3d = paths(p, seed)
        if (p, seed) in done and os.path.exists(t3d):
            print(f"\n[p={p} s={seed}] CSV 已有 ok 行且 3D 产物在，整档跳过",
                  flush=True)
            continue
        try:
            run_combo(p, seed)
        except SystemExit:
            raise
        except Exception as e:  # 单档异常不中止后续档
            print(f"[p={p} s={seed}] 未捕获异常：{e!r}，继续下一档", flush=True)
            append_row({"p": p, "q": round(p * p, 6), "seed": seed,
                        "status": "failed@exception",
                        "note": repr(e)[:200], "elapsed_s": 0})
    print(f"\n全部计划档位结束，总用时 {(time.time() - t_all) / 60:.1f} 分钟",
          flush=True)


if __name__ == "__main__":
    main()
