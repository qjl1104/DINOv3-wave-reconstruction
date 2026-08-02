# particle_processing/02b_detection_v2_loose.py
"""
v2 检测：放宽形状过滤，修复 02_particle_detection.py 的系统性漏检。

依据（2026-07-29 诊断，帧 0/125/.../875 双侧验证）：
  现参数 (A25/C.75/V.87/I.4, maxArea 300) 每帧左~153/右~177 点；
  本参数 (A15/C.55/V.75/I.3, maxArea 600) 每帧左~277/右~316 (+80%)，
  新增点局部峰值强度中位 209（噪声底 5），目检为偏暗/偏椭圆真泡沫。
  minDistBetweenBlobs 10→4 无影响（排除粘连间距因素）。
  更松 (A10/C.50/V.65/I.2, ~347/帧) 开始混入泡沫碎屑，不建议。

输出 detections_{left,right}_v2.pkl，不覆盖生产 pkl；
下游验证 OK 后，把 03 跟踪输入切换为 v2 再重跑链路。
"""
import os

import cv2
import glob
import numpy as np
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))

PARAMS = {
    'minArea': 15,
    'maxArea': 600,
    'minCircularity': 0.55,
    'minConvexity': 0.75,
    'minInertiaRatio': 0.3,
}


def detect_particles_blob(img, p):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = p['minArea']
    params.maxArea = p['maxArea']
    params.filterByCircularity = True
    params.minCircularity = p['minCircularity']
    params.filterByConvexity = True
    params.minConvexity = p['minConvexity']
    params.filterByInertia = True
    params.minInertiaRatio = p['minInertiaRatio']
    detector = cv2.SimpleBlobDetector_create(params)
    return [kp.pt for kp in detector.detect(img)]


def run_side(side):
    src = os.path.join(HERE, f"../data/preprocessed/{side}/*.png")
    files = sorted(glob.glob(src))
    out = []
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        out.append(detect_particles_blob(img, PARAMS) if img is not None else [])
        if (i + 1) % 250 == 0:
            print(f"  {side}: {i + 1}/{len(files)}", flush=True)
    counts = np.array([len(d) for d in out])
    print(f"{side}: mean={counts.mean():.0f} min={counts.min()} max={counts.max()}")
    dst = os.path.join(HERE, f"../data/detections/detections_{side}_v2.pkl")
    with open(dst, "wb") as fp:
        pickle.dump(out, fp)
    print(f"saved {dst}")


if __name__ == "__main__":
    run_side("left")
    run_side("right")
