# particle_processing/run_03_default.py
"""
默认参数（max_age=20 / min_hits=5 / dist_thresh=80，与 03_trajectory_tracking_2d_KLF_opus
一致）KLT 跟踪驱动，命令行指定输入输出（位置依次为：
det_left det_right out_left out_right prep_left prep_right，缺省用 v2 复现对照组）。
用法：../.venv_fs/Scripts/python.exe run_03_default.py [det_l det_r out_l out_r prep_l prep_r]
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tracker03", os.path.join(HERE, "03_trajectory_tracking_2d_KLF_opus.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 修正 pickle 类路径（与 run_03_tuned 相同）：使 rematch_rectified.py 的桩类可接管
for _name in dir(m):
    _obj = getattr(m, _name)
    if isinstance(_obj, type) and getattr(_obj, "__module__", None) == "tracker03":
        _obj.__module__ = "__main__"
        setattr(sys.modules["__main__"], _name, _obj)

det_l = sys.argv[1] if len(sys.argv) > 1 else "../data/detections/detections_left.pkl"
det_r = sys.argv[2] if len(sys.argv) > 2 else "../data/detections/detections_right.pkl"
out_l = sys.argv[3] if len(sys.argv) > 3 else "../data/trajectories/trajectories_2d_left_repro.pkl"
out_r = sys.argv[4] if len(sys.argv) > 4 else "../data/trajectories/trajectories_2d_right_repro.pkl"
prep_l = sys.argv[5] if len(sys.argv) > 5 else "../data/preprocessed/left/"
prep_r = sys.argv[6] if len(sys.argv) > 6 else "../data/preprocessed/right/"

tracking_params = {
    'max_age': 20,
    'min_hits_to_confirm': 5,
    'dist_thresh': 80.0
}

m.run_tracking_optimized(
    det_l, det_r, out_l, out_r,
    tracking_params, tracking_params,
    filter_type='optimized_ekf',
    show_live_video=False,
    preprocessed_dir_left=prep_l,
    preprocessed_dir_right=prep_r)
