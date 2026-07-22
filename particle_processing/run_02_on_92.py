# particle_processing/run_02_on_92.py
"""
在 opus_92 预处理输出（data/preprocessed_92/left|right）上跑 blob 检测，
输出 ../data/detections/detections_92_{left,right}.pkl。
默认沿用 02_particle_detection.py 的参数，可按需改。
用法：../.venv_fs/Scripts/python.exe run_02_on_92.py
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "det02", os.path.join(HERE, "02_particle_detection.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

params = {
    'minArea': 25,
    'maxArea': 300,
    'minCircularity': 0.75,
    'minConvexity': 0.87,
    'minInertiaRatio': 0.4
}

m.run_detection(
    "../data/preprocessed_92/left/", "../data/preprocessed_92/right/",
    "../data/detections/detections_92_left.pkl",
    "../data/detections/detections_92_right.pkl",
    params, params)
