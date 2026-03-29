import cv2
import numpy as np
import os
import glob

# --- 配置参数 ---
# 棋盘格内角点数 (需要根据图片确认，如果不确定，脚本会尝试以下组合)
# 师兄论文提及：宽18角点，高16角点（通常指方格数）
# 内角点数通常为 (cols-1, rows-1)
POSSIBLE_PATTERNS = [
    (17, 15), (15, 17),
    (18, 16), (16, 18),
    (17, 16), (16, 17)  # 容错
]


def load_calibration(calib_file):
    data = np.load(calib_file)
    return (data['Q'],
            data['map1_left'], data['map2_left'],
            data['map1_right'], data['map2_right'])


def find_right_image(left_path, right_dir):
    basename = os.path.basename(left_path)
    if '-' in basename:
        suffix = basename.split('-')[-1]
        pattern = os.path.join(right_dir, f"*{suffix}")
        candidates = glob.glob(pattern)
        if candidates:
            return candidates[0]
    return None


def enhance_image(gray):
    """
    图像增强：应对昏暗环境
    """
    # 1. CLAHE (限制对比度自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def find_corners_robust(img, patterns):
    """
    鲁棒的角点寻找函数
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 策略列表
    strategies = [
        ("Adaptive+Norm", gray, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK),
        ("CLAHE Enhanced", enhance_image(gray), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE),
        ("Default", gray, 0)
    ]

    for pat in patterns:
        for desc, img_input, flags in strategies:
            ret, corners = cv2.findChessboardCorners(img_input, pat, flags)
            if ret:
                return True, corners, pat

            # 降采样尝试
            small = cv2.resize(img_input, None, fx=0.5, fy=0.5)
            ret, corners = cv2.findChessboardCorners(small, pat, flags)
            if ret:
                return True, corners * 2.0, pat

    return False, None, None


def process_stereo_pair(l_path, r_path, m1l, m2l, m1r, m2r, Q):
    imgL = cv2.imread(l_path)
    imgR = cv2.imread(r_path)

    if imgL is None or imgR is None: return None

    # 1. 畸变校正
    imgL_rect = cv2.remap(imgL, m1l, m2l, cv2.INTER_LINEAR)
    imgR_rect = cv2.remap(imgR, m1r, m2r, cv2.INTER_LINEAR)

    # 2. 鲁棒寻找角点
    retL, cornersL, patL = find_corners_robust(imgL_rect, POSSIBLE_PATTERNS)
    if not retL: return None

    retR, cornersR, patR = find_corners_robust(imgR_rect, [patL])
    if not retR: return None

    # 3. 亚像素优化
    grayL = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

    # 4. 视差计算与 3D 重建
    ptsL = cornersL[:, 0, :]
    ptsR = cornersR[:, 0, :]

    # 简单校验 Y 轴对齐
    dy = np.abs(ptsL[:, 1] - ptsR[:, 1])
    if np.mean(dy) > 5.0: return None

    disp = ptsL[:, 0] - ptsR[:, 0]
    valid = (disp > 0.1) & (disp < 1000)
    if np.sum(valid) < 10: return None

    ptsL = ptsL[valid]
    disp = disp[valid]

    points_vec = np.stack([ptsL[:, 0], ptsL[:, 1], disp, np.ones_like(disp)], axis=1)
    homog_points = (Q @ points_vec.T).T
    points_3d = homog_points[:, :3] / homog_points[:, 3:4]

    return points_3d


def main():
    # --- 请修改这里的路径 ---
    calib_file = r"D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_from_matlab_full.npz"
    img_dir_left = r"D:\Research\wave_reconstruction_project\data\calibration_images\left"
    img_dir_right = r"D:\Research\wave_reconstruction_project\data\calibration_images\right"

    print(f"正在加载标定参数...")
    try:
        Q, m1l, m2l, m1r, m2r = load_calibration(calib_file)
    except Exception as e:
        print(f"加载标定文件失败: {e}")
        return

    left_images = sorted(glob.glob(os.path.join(img_dir_left, "*.bmp")))
    if not left_images:
        print("未找到 .bmp 图片，尝试查找 .jpg/.png ...")
        left_images = sorted(glob.glob(os.path.join(img_dir_left, "*.*")))

    print(f"找到 {len(left_images)} 张图片，开始增强扫描...")

    all_points_3d = []

    # 进度条效果
    total = len(left_images)
    for i, l_path in enumerate(left_images):
        r_path = find_right_image(l_path, img_dir_right)
        if r_path is None: continue

        pts = process_stereo_pair(l_path, r_path, m1l, m2l, m1r, m2r, Q)

        if pts is not None:
            all_points_3d.append(pts)
            print(f"[{i + 1}/{total}] 成功提取: {os.path.basename(l_path)} ({len(pts)} 点)")
        else:
            if i % 10 == 0:
                print(f"[{i + 1}/{total}] 处理中... (当前: {os.path.basename(l_path)})")

    if not all_points_3d:
        print("\n[错误] 所有图片均未提取到角点！")
        print("请检查是否光照太暗，或标定板格数设置错误。")
        return

    # --- 全局拟合 ---
    total_points = np.vstack(all_points_3d)
    # 过滤极值点
    mean_p = np.mean(total_points, axis=0)
    std_p = np.std(total_points, axis=0)
    mask = np.all(np.abs(total_points - mean_p) < 3 * std_p, axis=1)
    clean_points = total_points[mask]

    print(f"\n总共收集 {len(clean_points)} 个有效 3D 点，正在计算绝对水平面...")

    centroid = np.mean(clean_points, axis=0)
    centered = clean_points - centroid
    u, s, vt = np.linalg.svd(centered)
    normal = vt[2, :]

    if normal[1] < 0: normal = -normal

    print("\n" + "=" * 60)
    print("   【最终计算结果】 (请复制这些值到 inference 代码中)   ")
    print("=" * 60)
    print(f"CALIB_NORMAL = np.array([{normal[0]:.8f}, {normal[1]:.8f}, {normal[2]:.8f}])")
    print(f"CALIB_HEIGHT = {centroid[1]:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()