# run_full_pipeline.py
"""
波浪重建全链路端到端驱动（生产链，2026-07-29 切换为 v3 配方）：

  [0] v3 检测（02c_detection_v3_localmax.py：顶帽+局部极大值+分水岭，零形状先验）
      → data/detections/detections_{side}_v3.pkl
  [1] 外观辅助重捕跟踪（03b_tracker_appearance.py --no-foam：紧门 12px +
      NCC≥0.55 重捕；剔除 >1500px² 连通域来源的天然水沫检测点——连通域
      面积判据已经核验精度 100%，亮度判据被数据推翻未采用）
      → data/trajectories/trajectories_2d_{side}_v3nf.pkl
  [2] 跳切清洗（clean_tracks_jumpcut.py）
      → data/trajectories/trajectories_2d_{side}_v3nf_jumpcut.pkl
  [3] DINOv3 轨迹点描述子（compute_desc_tracks.py，已缓存且未过期则跳过）
      → DINOv3/desc_v3nftracks_{side}.pkl
  [4] DINO 辅助跨相机匹配（rematch_dino_v2.py --hung-only，匈牙利一对一最严口径）
      → data/trajectories/trajectories_3d_v3nf_hung_dino.pkl
  [5] 相干性评估（eval_tracks.py：η std / 0.79Hz 主峰率 / 互谱相位 c）
  [6] 方向先验 PINN（run_real_pinn.py）→ wave_modeling/real_run/pinn_real.pt
  [7] 最终可视化（final_visualize.py）
      → wave_modeling/real_run/final_result.png + final_field.npz

各阶段按 mtime 判断：输入比产物新才重跑，否则跳过。
首次切换时自动把旧生产产物（pinn_real.pt 等）备份到 real_run/_baseline_v1/。

用法：.venv_fs/Scripts/python.exe run_full_pipeline.py [--force]
"""

import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

DET_V3 = {s: os.path.join(ROOT, f"data/detections/detections_{s}_v3.pkl") for s in ("left", "right")}
TRAJ2D = {s: os.path.join(ROOT, f"data/trajectories/trajectories_2d_{s}_v3nf.pkl") for s in ("left", "right")}
TRAJ2D_JC = {s: os.path.join(ROOT, f"data/trajectories/trajectories_2d_{s}_v3nf_jumpcut.pkl") for s in ("left", "right")}
DESC = {s: os.path.join(ROOT, f"DINOv3/desc_v3nftracks_{s}.pkl") for s in ("left", "right")}
TRAJ_3D = os.path.join(ROOT, "data/trajectories/trajectories_3d_v3nf_hung_dino.pkl")
REAL_RUN = os.path.join(ROOT, "wave_modeling/real_run")
PINN_PT = os.path.join(REAL_RUN, "pinn_real.pt")
FINAL_PNG = os.path.join(REAL_RUN, "final_result.png")
FINAL_NPZ = os.path.join(REAL_RUN, "final_field.npz")
BASELINE_BAK = os.path.join(REAL_RUN, "_baseline_v1")

FORCE = "--force" in sys.argv


def newest_mtime(paths):
    return max(os.path.getmtime(p) for p in paths if os.path.exists(p)) if any(
        os.path.exists(p) for p in paths) else 0


def stale(outputs, inputs):
    """产物缺失或比输入旧 → 需要重跑。"""
    if FORCE:
        return True
    if not all(os.path.exists(o) for o in outputs):
        return True
    return min(os.path.getmtime(o) for o in outputs) < newest_mtime(inputs)


def run(stage, cmd, cwd=None):
    print(f"\n{'=' * 70}\n[阶段 {stage}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd or ROOT, env=ENV)
    if r.returncode != 0:
        sys.exit(f"[失败] 阶段 {stage} 退出码 {r.returncode}")
    print(f"[阶段 {stage}] 完成（{time.time() - t0:.0f}s）", flush=True)


def backup_baseline():
    """首次切换：备份旧生产产物（v1 基线，只备一次）。"""
    olds = ["pinn_real.pt", "final_result.png", "final_field.npz", "field_comparison.png"]
    if os.path.exists(PINN_PT) and not os.path.exists(BASELINE_BAK):
        os.makedirs(BASELINE_BAK)
        for f in olds:
            src = os.path.join(REAL_RUN, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(BASELINE_BAK, f))
        print(f"[备份] 旧生产产物已备份到 {BASELINE_BAK}")


def main():
    t_start = time.time()
    backup_baseline()

    imgs = {s: [os.path.join(ROOT, f"data/{s}_images", f)
                for f in os.listdir(os.path.join(ROOT, f"data/{s}_images"))] for s in ("left", "right")}

    # [0] v3 检测
    if stale(DET_V3.values(), sum(imgs.values(), [])):
        run("0/7 v3 检测（双侧）",
            [PY, "particle_processing/02c_detection_v3_localmax.py", "both"])
    else:
        print("[阶段 0/7] v3 检测已缓存且未过期，跳过")

    # [1] 外观跟踪（--no-foam：剔除 >1500px² 连通域来源的天然水沫检测点）
    if stale(TRAJ2D.values(), DET_V3.values()):
        run("1/7 外观辅助重捕跟踪（泡沫过滤）",
            [PY, "particle_processing/03b_tracker_appearance.py",
             DET_V3["left"], DET_V3["right"], TRAJ2D["left"], TRAJ2D["right"],
             "--no-foam"])
    else:
        print("[阶段 1/7] 2D 轨迹已缓存且未过期，跳过")

    # [2] 跳切清洗
    if stale(TRAJ2D_JC.values(), TRAJ2D.values()):
        run("2/7 跳切清洗",
            [PY, "particle_processing/clean_tracks_jumpcut.py",
             "--src", TRAJ2D["left"], TRAJ2D["right"],
             "--dst", TRAJ2D_JC["left"], TRAJ2D_JC["right"]])
    else:
        print("[阶段 2/7] 跳切清洗已缓存且未过期，跳过")

    # [3] DINO 描述子
    for s in ("left", "right"):
        if stale([DESC[s]], [TRAJ2D_JC[s]]):
            run(f"3/7 DINO 描述子（{s}）",
                [PY, "compute_desc_tracks.py", s, TRAJ2D_JC[s], DESC[s]],
                cwd=os.path.join(ROOT, "DINOv3"))
        else:
            print(f"[阶段 3/7] 描述子（{s}）已缓存且未过期，跳过")

    # [4] DINO 匹配（匈牙利一对一）
    if stale([TRAJ_3D], list(TRAJ2D_JC.values()) + list(DESC.values())):
        run("4/7 DINO 辅助跨相机匹配（--hung-only）",
            [PY, "rematch_dino_v2.py", TRAJ2D_JC["left"], TRAJ2D_JC["right"],
             DESC["left"], DESC["right"], TRAJ_3D, "--hung-only"],
            cwd=os.path.join(ROOT, "particle_processing"))
    else:
        print("[阶段 4/7] 3D 轨迹已缓存且未过期，跳过")

    # [5] 相干性评估（每次必跑，是裁判）
    run("5/7 相干性评估",
        [PY, "wave_modeling/eval_tracks.py", TRAJ_3D, "生产链 v3+外观跟踪"])

    # [6] PINN
    if stale([PINN_PT], [TRAJ_3D]):
        run("6/7 方向先验 PINN",
            [PY, "wave_modeling/run_real_pinn.py", TRAJ_3D])
    else:
        print("[阶段 6/7] PINN 已缓存且未过期，跳过")

    # [7] 最终可视化
    if stale([FINAL_PNG, FINAL_NPZ], [PINN_PT, TRAJ_3D]):
        run("7/7 最终可视化",
            [PY, "wave_modeling/final_visualize.py", TRAJ_3D])
    else:
        print("[阶段 7/7] 最终可视化已缓存且未过期，跳过")

    print(f"\n全链路完成，总用时 {(time.time() - t_start) / 60:.1f} 分钟。"
          f"成果：{FINAL_PNG}, {FINAL_NPZ}")


if __name__ == "__main__":
    main()
