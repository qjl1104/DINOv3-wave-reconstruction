# particle_processing/run_03_tuned.py
"""
实验B：调参版 KLT 跟踪（对照 03_trajectory_tracking_2d_KLF_opus 的默认参数）。
只放宽断档桥接（max_age 20→40，漏检桥接翻倍），关联门控 dist_thresh=80
不动（防止 ID 串接）；输出到新文件，不覆盖验证过的 *_optimized.pkl。
用法：../.venv_fs/Scripts/python.exe run_03_tuned.py
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tracker03", os.path.join(HERE, "03_trajectory_tracking_2d_KLF_opus.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# pickle 按 __module__ 记录类路径；importlib 加载时类属于 "tracker03"，
# 无法被 rematch_rectified.py 反序列化。把这些类的 __module__ 改为 "__main__"
# 并注入本脚本（__main__）命名空间使 pickle 校验通过；
# rematch（作为 __main__ 运行）里的同名桩类即可正常接管（与 *_optimized.pkl 机制相同）。
import sys as _sys
for _name in dir(m):
    _obj = getattr(m, _name)
    if isinstance(_obj, type) and getattr(_obj, "__module__", None) == "tracker03":
        _obj.__module__ = "__main__"
        setattr(_sys.modules["__main__"], _name, _obj)

tracking_params = {
    'max_age': 40,          # 默认 20 → 40：桥接更长漏检断档
    'min_hits_to_confirm': 5,
    'dist_thresh': 80.0     # 不动：关联精度是 ID 不串接的保证
}

m.run_tracking_optimized(
    "../data/detections/detections_left.pkl",
    "../data/detections/detections_right.pkl",
    "../data/trajectories/trajectories_2d_left_tuned.pkl",
    "../data/trajectories/trajectories_2d_right_tuned.pkl",
    tracking_params, tracking_params,
    filter_type='optimized_ekf',
    show_live_video=False,
    preprocessed_dir_left="../data/preprocessed/left/",
    preprocessed_dir_right="../data/preprocessed/right/")
