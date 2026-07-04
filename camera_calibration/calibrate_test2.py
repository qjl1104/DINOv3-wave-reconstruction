import cv2
import numpy as np
import glob
import os


def calibrate_stereo_camera(image_dir_left, image_dir_right, chessboard_size, square_size, params_save_path):
    """
    使用棋盘格图像校准立体相机系统（改进版本）。

    改进内容：
    1. 增强的角点检测算法
    2. 图像预处理
    3. 更严格的质量控制
    4. 详细的统计信息
    """
    # --- 1. 定义单个棋盘格图案的世界坐标 ---
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # --- 2. 准备用于存储校准点的列表 ---
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # --- 3. 加载图像路径 ---
    images_left = sorted(glob.glob(os.path.join(image_dir_left, '*.bmp')))
    images_right = sorted(glob.glob(os.path.join(image_dir_right, '*.bmp')))

    # --- 4. 基本的完整性检查 ---
    if not images_left or not images_right:
        print(f"错误：在 {image_dir_left} 或 {image_dir_right} 中未找到校准图像 (需要 .bmp 文件)")
        return None
    if len(images_left) != len(images_right):
        print("错误：左侧和右侧的校准图像数量不匹配。")
        return None

    print(f"找到了 {len(images_left)} 对用于校准的图像。")

    # --- 5. 获取图像尺寸 ---
    img_temp = cv2.imread(images_left[0])
    if img_temp is None:
        print(f"错误：无法读取第一张图像：{images_left[0]}")
        return None
    gray_shape = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY).shape[::-1]

    # --- 6. 改进的角点检测 ---
    # 角点检测的标志和参数
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

    # 统计信息
    stats = {
        'total_pairs': len(images_left),
        'left_detected': 0,
        'right_detected': 0,
        'both_detected': 0,
        'valid_pairs': 0
    }

    for i, (fname_left, fname_right) in enumerate(zip(images_left, images_right)):
        print(f"\n--- 正在处理第 {i + 1}/{len(images_left)} 对图像 ---")

        # 读取图像
        img_left = cv2.imread(fname_left)
        img_right = cv2.imread(fname_right)

        if img_left is None or img_right is None:
            print(f"  [错误] 无法读取图像对")
            continue

        # 转换为灰度图像
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 图像预处理（可选，用于改善角点检测）
        # 直方图均衡化
        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)

        # 高斯模糊去噪
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)

        # 使用改进的角点检测
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, chessboard_size, flags=chessboard_flags)
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, chessboard_size, flags=chessboard_flags)

        # 更新统计信息
        stats['left_detected'] += ret_left
        stats['right_detected'] += ret_right
        stats['both_detected'] += (ret_left and ret_right)

        # --- 可视化模块 ---
        img_left_draw = img_left.copy()
        img_right_draw = img_right.copy()

        if ret_left:
            print(f"  [成功] 在左侧图像中找到角点: {os.path.basename(fname_left)}")
            cv2.drawChessboardCorners(img_left_draw, chessboard_size, corners_left, ret_left)
        else:
            print(f"  [失败] 未在左侧图像中找到角点: {os.path.basename(fname_left)}")

        if ret_right:
            print(f"  [成功] 在右侧图像中找到角点: {os.path.basename(fname_right)}")
            cv2.drawChessboardCorners(img_right_draw, chessboard_size, corners_right, ret_right)
        else:
            print(f"  [失败] 未在右侧图像中找到角点: {os.path.basename(fname_right)}")

        # 如果两边都找到了角点，进行质量检查
        if ret_left and ret_right:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            # 质量检查：计算角点的分布质量
            quality_left = assess_corner_quality(corners_left_subpix, gray_shape)
            quality_right = assess_corner_quality(corners_right_subpix, gray_shape)

            print(f"  -> 左侧角点质量: {quality_left:.3f}, 右侧角点质量: {quality_right:.3f}")

            # 只保留高质量的角点
            if quality_left > 0.1 and quality_right > 0.1:  # 阈值可调整
                print("  -> 这对图像可用于校准。")
                objpoints.append(objp)
                imgpoints_left.append(corners_left_subpix)
                imgpoints_right.append(corners_right_subpix)
                stats['valid_pairs'] += 1
            else:
                print("  -> 角点质量不佳，跳过此对图像。")
        else:
            print("  -> 这对图像无效，将被跳过。")

        # 显示图像
        display_img = np.hstack([img_left_draw, img_right_draw])
        h, w, _ = display_img.shape
        scale_factor = 1280 / w
        if scale_factor < 1:
            display_img = cv2.resize(display_img, (int(w * scale_factor), int(h * scale_factor)))

        cv2.imshow('Chessboard Corner Detection - Press any key to continue', display_img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # 按'q'退出
            break

    cv2.destroyAllWindows()

    # --- 7. 打印统计信息 ---
    print(f"\n=== 角点检测统计 ===")
    print(f"总图像对数: {stats['total_pairs']}")
    print(f"左侧成功检测: {stats['left_detected']} ({stats['left_detected'] / stats['total_pairs'] * 100:.1f}%)")
    print(f"右侧成功检测: {stats['right_detected']} ({stats['right_detected'] / stats['total_pairs'] * 100:.1f}%)")
    print(f"双侧都检测到: {stats['both_detected']} ({stats['both_detected'] / stats['total_pairs'] * 100:.1f}%)")
    print(f"有效校准对数: {stats['valid_pairs']} ({stats['valid_pairs'] / stats['total_pairs'] * 100:.1f}%)")

    # --- 8. 执行校准 ---
    if not objpoints:
        print("\n错误：没有找到足够的有效图像对进行校准。")
        return None

    if len(objpoints) < 10:
        print(f"\n警告：只找到 {len(objpoints)} 对有效图像，建议至少使用10对以上。")

    print(f"\n使用 {len(objpoints)} 对有效的图像进行校准。")

    # 改进的校准参数
    calibration_flags = (cv2.CALIB_RATIONAL_MODEL +
                         cv2.CALIB_THIN_PRISM_MODEL +
                         cv2.CALIB_TILTED_MODEL)

    # 单独校准每个相机
    print("正在进行初始的单相机校准...")
    ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_shape, None, None, flags=calibration_flags)
    ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_shape, None, None, flags=calibration_flags)

    print(f"左相机校准RMS误差: {ret_l:.4f}")
    print(f"右相机校准RMS误差: {ret_r:.4f}")

    # 立体校准
    print("正在进行立体校准...")
    stereocalib_flags = (cv2.CALIB_FIX_INTRINSIC +
                         cv2.CALIB_RATIONAL_MODEL +
                         cv2.CALIB_THIN_PRISM_MODEL)
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

    ret_stereo, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K_l, D_l, K_r, D_r, gray_shape,
        criteria=stereocalib_criteria,
        flags=stereocalib_flags
    )

    print(f"立体校准成功。RMS重投影误差: {ret_stereo:.4f}")

    # --- 9. 校正并保存参数 ---
    print("正在校正相机并保存参数...")
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, gray_shape, R, T, alpha=0
    )

    map1_left, map2_left = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, gray_shape, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, gray_shape, cv2.CV_16SC2)

    if not os.path.exists(params_save_path):
        os.makedirs(params_save_path)

    save_file_path = os.path.join(params_save_path, "stereo_calib_params.npz")
    np.savez(save_file_path,
             K_left=K_left, D_left=D_left, K_right=K_right, D_right=D_right,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             roi_left=roi_left, roi_right=roi_right,
             image_size=gray_shape,
             map1_left=map1_left, map2_left=map2_left,
             map1_right=map1_right, map2_right=map2_right,
             # 保存校准质量信息
             rms_stereo=ret_stereo,
             rms_left=ret_l,
             rms_right=ret_r,
             num_valid_pairs=len(objpoints))

    print(f"校准参数已保存至 {save_file_path}")

    # 保存校准报告
    save_calibration_report(save_file_path.replace('.npz', '_report.txt'),
                            stats, ret_l, ret_r, ret_stereo, len(objpoints))

    return True


def assess_corner_quality(corners, image_shape):
    """
    评估角点检测的质量
    返回值越高表示质量越好
    """
    if corners is None or len(corners) == 0:
        return 0.0

    # 计算角点分布的覆盖范围
    corners_2d = corners.reshape(-1, 2)
    x_range = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
    y_range = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])

    # 归一化到图像尺寸
    coverage = (x_range * y_range) / (image_shape[0] * image_shape[1])

    return coverage


def save_calibration_report(report_path, stats, rms_left, rms_right, rms_stereo, valid_pairs):
    """保存校准报告"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 立体相机校准报告 ===\n\n")
        f.write(f"总图像对数: {stats['total_pairs']}\n")
        f.write(
            f"左侧成功检测: {stats['left_detected']} ({stats['left_detected'] / stats['total_pairs'] * 100:.1f}%)\n")
        f.write(
            f"右侧成功检测: {stats['right_detected']} ({stats['right_detected'] / stats['total_pairs'] * 100:.1f}%)\n")
        f.write(
            f"双侧都检测到: {stats['both_detected']} ({stats['both_detected'] / stats['total_pairs'] * 100:.1f}%)\n")
        f.write(f"有效校准对数: {stats['valid_pairs']} ({stats['valid_pairs'] / stats['total_pairs'] * 100:.1f}%)\n\n")

        f.write("=== 校准精度 ===\n")
        f.write(f"左相机RMS误差: {rms_left:.4f} 像素\n")
        f.write(f"右相机RMS误差: {rms_right:.4f} 像素\n")
        f.write(f"立体校准RMS误差: {rms_stereo:.4f} 像素\n\n")

        f.write("=== 质量评估 ===\n")
        if rms_stereo < 0.5:
            f.write("校准质量: 优秀\n")
        elif rms_stereo < 1.0:
            f.write("校准质量: 良好\n")
        elif rms_stereo < 2.0:
            f.write("校准质量: 一般\n")
        else:
            f.write("校准质量: 需要改进\n")

        if valid_pairs < 10:
            f.write("建议: 增加更多高质量的校准图像\n")


if __name__ == '__main__':
    # --- 参数设置 ---
    left_calib_img_dir = "../data/calibration_images/left"
    right_calib_img_dir = "../data/calibration_images/right"
    output_params_dir = "params"

    chessboard_corners_cols = 18
    chessboard_corners_rows = 16
    square_side_length_mm = 45

    print("开始改进的立体校准流程...")
    calibrate_stereo_camera(
        left_calib_img_dir,
        right_calib_img_dir,
        (chessboard_corners_cols, chessboard_corners_rows),
        square_side_length_mm,
        output_params_dir
    )
    print("流程结束。")