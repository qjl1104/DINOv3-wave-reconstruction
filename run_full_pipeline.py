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
PY = sys.executable  # 用启动本脚本的解释器，不再硬编码 .venv_fs 路径
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

DESC_L = os.path.join(ROOT, "DINOv3/desc_v2tracks_left.pkl")
DESC_R = os.path.join(ROOT, "DINOv3/desc_v2tracks_right.pkl")
TRAJ_2D_L = os.path.join(ROOT, "data/trajectories/trajectories_2d_left_optimized.pkl")
TRAJ_2D_R = os.path.join(ROOT, "data/trajectories/trajectories_2d_right_optimized.pkl")
TRAJ_3D = os.path.join(ROOT, "data/trajectories/trajectories_3d_v2_dino.pkl")
PINN_PT = os.path.join(ROOT, "wave_modeling/real_run/pinn_real.pt")
FINAL_PNG = os.path.join(ROOT, "wave_modeling/real_run/final_result.png")
FINAL_NPZ = os.path.join(ROOT, "wave_modeling/real_run/final_field.npz")

# 各阶段预期产出 → (路径, 最小字节数)，退出码之外再做产出校验
# （阶段 3 相干性评估只打印不落盘，无产出可校验）
STAGE_OUTPUTS = {
    "2/5": [(TRAJ_3D, 1024)],  # rematch 产出 pkl 必须非平凡
    "4/5": [(PINN_PT, 1)],
    "5/5": [(FINAL_PNG, 1), (FINAL_NPZ, 1)],
}


def run(stage, cmd, cwd=None):
    print(f"\n{'=' * 70}\n[阶段 {stage}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd or ROOT, env=ENV)
    if r.returncode != 0:
        print(f"[失败] 阶段 {stage} 退出码 {r.returncode}", flush=True)
        sys.exit(r.returncode)
    for out, min_bytes in STAGE_OUTPUTS.get(stage.split()[0], []):
        if not (os.path.exists(out) and os.path.getsize(out) >= min_bytes):
            sys.exit(f"[失败] 阶段 {stage} 退出码为 0 但预期产出缺失或过小：{out}，链路中止")
    print(f"[阶段 {stage}] 完成（{time.time() - t0:.0f}s）", flush=True)


def main():
    force_desc = "--force-desc" in sys.argv
    for p in [TRAJ_2D_L, TRAJ_2D_R]:
        if not os.path.exists(p):
            sys.exit(f"[缺少输入] {p}（canonical 2D 轨迹，不可再生成）")

    t_start = time.time()
    desc_reason = None
    if force_desc:
        desc_reason = "--force-desc 指定重算"
    elif not (os.path.exists(DESC_L) and os.path.exists(DESC_R)):
        desc_reason = "描述子缓存缺失"
    else:
        # 过期检查：描述子由 canonical 2D 轨迹 pkl 派生，输入比缓存新 → 重算
        for traj, desc in [(TRAJ_2D_L, DESC_L), (TRAJ_2D_R, DESC_R)]:
            if os.path.getmtime(traj) > os.path.getmtime(desc):
                desc_reason = (f"输入 {os.path.basename(traj)} 比缓存 "
                               f"{os.path.basename(desc)} 新")
                break
    if desc_reason:
        print(f"[阶段 1/5] 重算 DINO 描述子：{desc_reason}")
        run("1a/5 DINO 描述子（左）",
            [PY, "compute_desc_tracks.py", "left"], cwd=os.path.join(ROOT, "DINOv3"))
        run("1b/5 DINO 描述子（右）",
            [PY, "compute_desc_tracks.py", "right"], cwd=os.path.join(ROOT, "DINOv3"))
    else:
        print("[阶段 1/5] DINO 描述子已缓存且未过期，跳过（--force-desc 可重算）")

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
