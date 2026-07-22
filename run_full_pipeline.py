# run_full_pipeline.py
"""
波浪重建全链路端到端驱动（生产链）：

  [1] DINOv3 轨迹点描述子（compute_desc_tracks.py，已缓存则跳过）
  [2] DINO 辅助跨相机匹配（rematch_dino_v2.py）
      → data/trajectories/trajectories_3d_v2_dino.pkl
  [3] 相干性评估（eval_tracks.py：η std / 0.79Hz 主峰率 / 互谱相位 c）
  [4] 方向先验 PINN（run_real_pinn.py：互谱测向 + 旋转 + 固定 c=1976）
      → wave_modeling/real_run/pinn_real.pt
  [5] 最终可视化（final_visualize.py）
      → wave_modeling/real_run/final_result.png + final_field.npz

前置输入（canonical，勿删）：data/trajectories/trajectories_2d_*_optimized.pkl、
data/left_images、data/right_images、DINOv3/dinov3-base-model、标定参数。

用法：.venv_fs/Scripts/python.exe run_full_pipeline.py [--force-desc]
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv_fs", "Scripts", "python.exe")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

DESC_L = os.path.join(ROOT, "DINOv3/desc_v2tracks_left.pkl")
DESC_R = os.path.join(ROOT, "DINOv3/desc_v2tracks_right.pkl")
TRAJ_2D_L = os.path.join(ROOT, "data/trajectories/trajectories_2d_left_optimized.pkl")
TRAJ_2D_R = os.path.join(ROOT, "data/trajectories/trajectories_2d_right_optimized.pkl")
TRAJ_3D = os.path.join(ROOT, "data/trajectories/trajectories_3d_v2_dino.pkl")


def run(stage, cmd, cwd=None):
    print(f"\n{'=' * 70}\n[阶段 {stage}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd or ROOT, env=ENV)
    if r.returncode != 0:
        print(f"[失败] 阶段 {stage} 退出码 {r.returncode}", flush=True)
        sys.exit(r.returncode)
    print(f"[阶段 {stage}] 完成（{time.time() - t0:.0f}s）", flush=True)


def main():
    force_desc = "--force-desc" in sys.argv
    for p in [TRAJ_2D_L, TRAJ_2D_R]:
        if not os.path.exists(p):
            sys.exit(f"[缺少输入] {p}（canonical 2D 轨迹，不可再生成）")

    t_start = time.time()
    if force_desc or not (os.path.exists(DESC_L) and os.path.exists(DESC_R)):
        run("1a/5 DINO 描述子（左）",
            [PY, "compute_desc_tracks.py", "left"], cwd=os.path.join(ROOT, "DINOv3"))
        run("1b/5 DINO 描述子（右）",
            [PY, "compute_desc_tracks.py", "right"], cwd=os.path.join(ROOT, "DINOv3"))
    else:
        print("[阶段 1/5] DINO 描述子已缓存，跳过（--force-desc 可重算）")

    run("2/5 DINO 辅助跨相机匹配",
        [PY, "rematch_dino_v2.py"], cwd=os.path.join(ROOT, "particle_processing"))

    run("3/5 相干性评估",
        [PY, "wave_modeling/eval_tracks.py", TRAJ_3D, "全链路产出 v2_dino"])

    run("4/5 方向先验 PINN",
        [PY, "wave_modeling/run_real_pinn.py", TRAJ_3D])

    run("5/5 最终可视化",
        [PY, "wave_modeling/final_visualize.py", TRAJ_3D])

    print(f"\n全链路完成，总用时 {(time.time() - t_start) / 60:.1f} 分钟。"
          f"成果：wave_modeling/real_run/final_result.png, final_field.npz")


if __name__ == "__main__":
    main()
