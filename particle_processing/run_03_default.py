# particle_processing/run_03_default.py
"""
默认参数（max_age=20 / min_hits=5 / dist_thresh=80，与 03_trajectory_tracking_2d_KLF_opus
一致）KLT 跟踪驱动。位置参数依次为 det_left det_right out_left out_right prep_left
prep_right（缺省用 v2 复现对照组；路径锚定脚本所在目录，与 CWD 无关）。
跟踪参数可用 --max-age/--min-hits/--dist-thresh 覆盖（run_03_tuned.py 即如此复用本脚本）。
用法：../.venv_fs/Scripts/python.exe run_03_default.py [det_l det_r out_l out_r prep_l prep_r]
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(argv=None):
    ap = argparse.ArgumentParser(description="默认参数 KLT 跟踪驱动")
    ap.add_argument("det_l", nargs="?",
                    default=os.path.join(ROOT, "data/detections/detections_left.pkl"))
    ap.add_argument("det_r", nargs="?",
                    default=os.path.join(ROOT, "data/detections/detections_right.pkl"))
    ap.add_argument("out_l", nargs="?",
                    default=os.path.join(ROOT, "data/trajectories/trajectories_2d_left_repro.pkl"))
    ap.add_argument("out_r", nargs="?",
                    default=os.path.join(ROOT, "data/trajectories/trajectories_2d_right_repro.pkl"))
    ap.add_argument("prep_l", nargs="?",
                    default=os.path.join(ROOT, "data/preprocessed/left/"))
    ap.add_argument("prep_r", nargs="?",
                    default=os.path.join(ROOT, "data/preprocessed/right/"))
    ap.add_argument("--max-age", type=int, default=20)
    ap.add_argument("--min-hits", type=int, default=5)
    ap.add_argument("--dist-thresh", type=float, default=80.0)
    args = ap.parse_args(argv)

    spec = importlib.util.spec_from_file_location(
        "tracker03", os.path.join(HERE, "03_trajectory_tracking_2d_KLF_opus.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    # pickle 按 __module__ 记录类路径；importlib 加载时类属于 "tracker03"，
    # 无法被 rematch_rectified.py 反序列化。把这些类的 __module__ 改为 "__main__"
    # 并注入 __main__ 命名空间使 pickle 校验通过；
    # rematch（作为 __main__ 运行）里的同名桩类即可正常接管（与 *_optimized.pkl 机制相同）。
    for _name in dir(m):
        _obj = getattr(m, _name)
        if isinstance(_obj, type) and getattr(_obj, "__module__", None) == "tracker03":
            _obj.__module__ = "__main__"
            setattr(sys.modules["__main__"], _name, _obj)

    tracking_params = {
        'max_age': args.max_age,
        'min_hits_to_confirm': args.min_hits,
        'dist_thresh': args.dist_thresh
    }

    m.run_tracking_optimized(
        args.det_l, args.det_r, args.out_l, args.out_r,
        tracking_params, tracking_params,
        filter_type='optimized_ekf',
        show_live_video=False,
        preprocessed_dir_left=args.prep_l,
        preprocessed_dir_right=args.prep_r)


if __name__ == "__main__":
    main()
