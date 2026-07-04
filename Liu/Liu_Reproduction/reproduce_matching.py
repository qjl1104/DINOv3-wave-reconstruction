import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 将 SuperGlue 库加入路径
sys.path.append(os.path.join(os.getcwd(), 'SuperGluePretrainedNetwork'))
from models.matching import Matching
from models.utils import frame2tensor

# ================= 配置参数 (根据论文复现) =================
CONFIG = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    },
    'data_path_L': r'D:\Research\wave_reconstruction_project\data\lresult',
    'data_path_R': r'D:\Research\wave_reconstruction_project\data\rresult',
    'output_dir': r'D:\Research\wave_reconstruction_project\output_matches',
    'start_idx': 1,  # 从第1张开始
    'count': 5       # 先测试5张，跑通后再跑1000张
}

# 检查是否有GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running inference on device: {device}")

# 初始化 SuperPoint + SuperGlue
matching = Matching(CONFIG).eval().to(device)

def preprocess_image_paper_method(img_path):
    """
    复现论文 3.2.2 节提到的预处理：
    1. 灰度值 < 150 置为 0 (去泡沫干扰) 
    2. 膨胀处理 (扩大特征点) 
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot read {img_path}")
        return None

    # 1. 阈值处理
    _, thresh_img = cv2.threshold(img, 150, 255, cv2.THRESH_TOZERO)

    # 2. 膨胀处理 (3x3 卷积核)
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
    
    return dilated_img

def filter_matches_epipolar(kpts0, kpts1, matches, threshold=5.0):
    """
    复现论文 2.2.4 节提到的极线约束：
    垂直方向上的位置差异不得超过 5 个像素 。
    """
    valid_matches = []
    mkpts0, mkpts1 = [], []

    # matches[i] > -1 表示第 i 个点有匹配
    for i in range(len(matches)):
        if matches[i] > -1:
            idx1 = matches[i]
            pt0 = kpts0[i]
            pt1 = kpts1[idx1]
            
            # 计算垂直视差 (y轴差异)
            y_diff = abs(pt0[1] - pt1[1])
            
            # 极线约束过滤
            if y_diff <= threshold:
                valid_matches.append((i, idx1))
                mkpts0.append(pt0)
                mkpts1.append(pt1)
    
    return np.array(mkpts0), np.array(mkpts1)

def run_pipeline():
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

    for i in range(CONFIG['start_idx'], CONFIG['start_idx'] + CONFIG['count']):
        # 构造文件名 (例如 lresult0001.bmp)
        filename_l = f"lresult{i:04d}.bmp"
        filename_r = f"rresult{i:04d}.bmp"
        path_l = os.path.join(CONFIG['data_path_L'], filename_l)
        path_r = os.path.join(CONFIG['data_path_R'], filename_r)

        # 1. 预处理
        img0_raw = preprocess_image_paper_method(path_l)
        img1_raw = preprocess_image_paper_method(path_r)
        
        if img0_raw is None or img1_raw is None: continue

        # 2. 转为 Tensor 输入网络
        frame0 = frame2tensor(img0_raw, device)
        frame1 = frame2tensor(img1_raw, device)

        # 3. 推理 (SuperPoint + SuperGlue)
        with torch.no_grad():
            pred = matching({'image0': frame0, 'image1': frame1})
            
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        
        # 4. 极线约束过滤
        mkpts0, mkpts1 = filter_matches_epipolar(kpts0, kpts1, matches)
        
        print(f"Frame {i}: Extracted {len(kpts0)}/{len(kpts1)} keypoints. "
              f"Matched {len(mkpts0)} pairs after epipolar constraint.")

        # 5. 可视化保存 (验证是否出现类似论文图4.3的平行匹配线)
        viz_img = cv2.hconcat([img0_raw, img1_raw])
        viz_img = cv2.cvtColor(viz_img, cv2.COLOR_GRAY2BGR)
        
        for pt0, pt1 in zip(mkpts0, mkpts1):
            # 绘制匹配线
            pt0 = (int(pt0[0]), int(pt0[1]))
            pt1 = (int(pt1[0] + img0_raw.shape[1]), int(pt1[1])) # 右图x坐标偏移
            cv2.line(viz_img, pt0, pt1, (0, 255, 0), 1)
            cv2.circle(viz_img, pt0, 2, (0, 0, 255), -1)
            cv2.circle(viz_img, pt1, 2, (0, 0, 255), -1)

        save_path = os.path.join(CONFIG['output_dir'], f"match_{i:04d}.jpg")
        cv2.imwrite(save_path, viz_img)

    print(f"Done! Results saved to {CONFIG['output_dir']}")

if __name__ == '__main__':
    run_pipeline()