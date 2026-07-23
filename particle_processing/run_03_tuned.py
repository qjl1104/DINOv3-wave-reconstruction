# particle_processing/run_03_tuned.py
"""
实验B：调参版 KLT 跟踪（对照 03_trajectory_tracking_2d_KLF_opus 的默认参数）。
只放宽断档桥接（max_age 20→40，漏检桥接翻倍），关联门控 dist_thresh=80
不动（防止 ID 串接）；输出到新文件，不覆盖验证过的 *_optimized.pkl。
实现已并入 run_03_default.py，本脚本只是传调参值的薄封装（文件名保留，README 引用）。
用法：../.venv_fs/Scripts/python.exe run_03_tuned.py
"""

import os

import run_03_default

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

if __name__ == "__main__":
    run_03_default.main([
        os.path.join(ROOT, "data/detections/detections_left.pkl"),
        os.path.join(ROOT, "data/detections/detections_right.pkl"),
        os.path.join(ROOT, "data/trajectories/trajectories_2d_left_tuned.pkl"),
        os.path.join(ROOT, "data/trajectories/trajectories_2d_right_tuned.pkl"),
        os.path.join(ROOT, "data/preprocessed/left/"),
        os.path.join(ROOT, "data/preprocessed/right/"),
        "--max-age", "40",  # 默认 20 → 40：桥接更长漏检断档
        # dist_thresh 不动：关联精度是 ID 不串接的保证
    ])
