import cv2
import torch
import numpy as np
import os
import sys
from scipy.optimize import leastsq

# ===============================================================================
# 0. 环境配置
# ===============================================================================
sys.path.append(os.path.join(os.getcwd(), 'SuperGluePretrainedNetwork'))

try:
    from models.matching import Matching
    from models.utils import frame2tensor
except ImportError:
    print("错误: 找不到 SuperGlue 模型文件。")
    sys.exit(1)

# ===============================================================================
# 1. 核心标定参数
# ===============================================================================
K_L = np.array([[3937.20908, 0, 1349.85569], [0, 3936.40700, 952.73155], [0, 0, 1]])
K_R = np.array([[3925.72662, 0, 1278.88564], [0, 3924.51774, 886.81188], [0, 0, 1]])

# 旋转向量 (保持原始值)
r_vec = np.array([-0.03702, 0.25002, 0.1387]) 
R_mat, _ = cv2.Rodrigues(r_vec)

# 平移向量 T
T_vec = np.array([[-1397.67526], [-141.94746], [153.43894]])

# ===============================================================================
# 2. 项目配置
# ===============================================================================
REAL_WORLD_SIZE = 3000.0   
GRID_SIZE = 256            

CONFIG = {
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 2048},
    'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2},
    'data_path_L': r'D:\Research\wave_reconstruction_project\data\lresult',
    'data_path_R': r'D:\Research\wave_reconstruction_project\data\rresult',
    'output_dir': r'D:\Research\wave_reconstruction_project\output_depth_maps',
    'start_idx': 1,
    'count': 1000  # 全量运行
}

# ===============================================================================
# 3. 功能函数
# ===============================================================================

def get_scaled_parameters(img_width):
    CALIB_WIDTH = 2560.0
    scale_factor = img_width / CALIB_WIDTH
    if abs(scale_factor - 1.0) < 0.05:
        return K_L, K_R, 1.0
    print(f"  [Info] Image resized (width={img_width}). Scaling focal length by {scale_factor:.2f}")
    K_L_new, K_R_new = K_L.copy(), K_R.copy()
    K_L_new[:2, :] *= scale_factor
    K_R_new[:2, :] *= scale_factor
    return K_L_new, K_R_new, scale_factor

def detrend_wave_height(X, Y, Z):
    # 去趋势：拟合平面并减去
    def plane_error(params, x, y, z):
        a, b, c = params
        return z - (a * x + b * y + c)
    
    p0 = [0, 0, np.mean(Z)]
    params, _ = leastsq(plane_error, p0, args=(X, Y, Z))
    a, b, c = params
    
    # 计算相对波高 (取负号，使波峰为正值)
    Z_rel = -(Z - (a * X + b * Y + c))
    return Z_rel

def robust_triangulation(mkpts0, mkpts1, P1, P2):
    pts_l = mkpts0.T
    pts_r = mkpts1.T
    points_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    return points_3d.T 

def generate_grid_map(points_3d, save_path):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    if points_3d.shape[0] < 10: return

    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # 1. 物理距离过滤 (针对 4.5m 左右的高度)
    z_mean = np.mean(Z)
    if i % 100 == 0:
        print(f"  [Stats] Raw Z mean: {z_mean:.1f} mm")
    
    # 允许较大的波动范围，防止切掉浪尖
    valid_mask = np.abs(Z - z_mean) < 1500 
    X, Y, Z = X[valid_mask], Y[valid_mask], Z[valid_mask]
    if len(Z) < 10: return

    # 2. 去趋势
    Z_rel = detrend_wave_height(X, Y, Z)

    # 3. 归一化 (振幅 +/- 60mm)
    wave_range = 60.0 
    Z_norm = np.clip((Z_rel + wave_range) / (2 * wave_range) * 255, 0, 255).astype(np.uint8)

    # 4. 投影
    x_center, y_center = np.mean(X), np.mean(Y)
    u_grid = ((X - x_center + REAL_WORLD_SIZE/2) / REAL_WORLD_SIZE * GRID_SIZE).astype(np.int32)
    v_grid = ((Y - y_center + REAL_WORLD_SIZE/2) / REAL_WORLD_SIZE * GRID_SIZE).astype(np.int32)

    valid_uv = (u_grid >= 0) & (u_grid < GRID_SIZE) & (v_grid >= 0) & (v_grid < GRID_SIZE)
    grid[v_grid[valid_uv], u_grid[valid_uv]] = Z_norm[valid_uv]

    cv2.imwrite(save_path, grid)

# ===============================================================================
# 4. 主程序
# ===============================================================================

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    matching = Matching(CONFIG).eval().to(device)
    if not os.path.exists(CONFIG['output_dir']): os.makedirs(CONFIG['output_dir'])

    # 初始化参数
    first_img_path = os.path.join(CONFIG['data_path_L'], f"lresult{CONFIG['start_idx']:04d}.bmp")
    if not os.path.exists(first_img_path):
        print(f"Error: Cannot find {first_img_path}")
        return
    
    sample_img = cv2.imread(first_img_path)
    K_L_final, K_R_final, _ = get_scaled_parameters(sample_img.shape[1])
    
    # --- 关键修正：恢复 "Right-to-Left" 模式 ---
    # 这能保证 Z 值在正确的 4500mm 范围
    # P1 (主, Origin) = 右相机
    # P2 (副, Target) = 左相机
    P1 = K_R_final @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_L_final @ np.hstack((R_mat, T_vec))

    print(f"Start processing {CONFIG['count']} frames...")
    print(f"Output directory: {CONFIG['output_dir']}")

    global i
    for i in range(CONFIG['start_idx'], CONFIG['start_idx'] + CONFIG['count']):
        filename_l = f"lresult{i:04d}.bmp"
        filename_r = f"rresult{i:04d}.bmp"
        path_l = os.path.join(CONFIG['data_path_L'], filename_l)
        path_r = os.path.join(CONFIG['data_path_R'], filename_r)

        # 读取顺序：img0=Right, img1=Left
        img0 = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
        
        if img0 is None: continue
        
        # 预处理
        _, img0_th = cv2.threshold(img0, 150, 255, cv2.THRESH_TOZERO)
        _, img1_th = cv2.threshold(img1, 150, 255, cv2.THRESH_TOZERO)
        kernel = np.ones((3,3), np.uint8)
        img0_proc = cv2.dilate(img0_th, kernel, iterations=1)
        img1_proc = cv2.dilate(img1_th, kernel, iterations=1)

        # 匹配
        frame0 = frame2tensor(img0_proc, device)
        frame1 = frame2tensor(img1_proc, device)
        with torch.no_grad():
            pred = matching({'image0': frame0, 'image1': frame1})
            
        matches = pred['matches0'][0].cpu().numpy()
        valid = np.where(matches > -1)[0]
        mkpts0 = pred['keypoints0'][0].cpu().numpy()[valid]
        mkpts1 = pred['keypoints1'][0].cpu().numpy()[matches[valid]]

        if len(mkpts0) < 10: continue

        # 重建与保存
        points_3d = robust_triangulation(mkpts0, mkpts1, P1, P2)
        save_name = os.path.join(CONFIG['output_dir'], f"depth_{i:04d}.png")
        generate_grid_map(points_3d, save_name)
        
        if (i - CONFIG['start_idx']) % 50 == 0:
            print(f"Processed {i} frames. Last cloud size: {len(points_3d)}")

    print("All done! Dataset generation complete.")

if __name__ == '__main__':
    run()