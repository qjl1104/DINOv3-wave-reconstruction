import cv2
import numpy as np
import glob
import os


def calibrate_stereo_camera(image_dir_left, image_dir_right, chessboard_size, square_size, params_save_path):
    """
    使用棋盘格图像校准立体相机系统。

    此函数现在包含一个可视化步骤，以帮助调试角点检测。
    它将显示每一对图像并绘制检测到的角点。
    """
    # --- 1. 定义单个棋盘格图案的世界坐标 ---
    # 创建棋盘格角点的 (X, Y, Z) 坐标。Z为0，因为图案是平面的。
    # 这个网格是为*内部角点*的数量创建的。
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # --- 2. 准备用于存储校准点的列表 ---
    objpoints = []  # 真实世界空间中的3D点
    imgpoints_left = []  # 左相机图像平面中的2D点
    imgpoints_right = []  # 右相机图像平面中的2D点

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
    # 我们假设所有图像的尺寸都相同。
    img_temp = cv2.imread(images_left[0])
    if img_temp is None:
        print(f"错误：无法读取第一张图像：{images_left[0]}")
        return None
    gray_shape = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY).shape[::-1]  # (宽度, 高度)

    # --- 6. 在每对图像中查找角点 (带可视化功能) ---
    for i, (fname_left, fname_right) in enumerate(zip(images_left, images_right)):
        print(f"\n--- 正在处理第 {i + 1}/{len(images_left)} 对图像 ---")
        img_left = cv2.imread(fname_left)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.imread(fname_right)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 在两个图像中查找棋盘格角点
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

        # --- 调试可视化模块 ---
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

        # 并排显示图像
        display_img = np.hstack([img_left_draw, img_right_draw])
        # 如有必要，调整大小以更好地适应屏幕
        h, w, _ = display_img.shape
        scale_factor = 1280 / w  # 目标是1280像素宽的窗口
        if scale_factor < 1:
            display_img = cv2.resize(display_img, (int(w * scale_factor), int(h * scale_factor)))

        # Changed UI title to avoid encoding issues
        cv2.imshow('Chessboard Corner Detection (Left | Right) - Press any key to continue', display_img)
        cv2.waitKey(0)  # 等待按键以继续处理下一张图像
        # --- 调试可视化模块结束 ---

        # 如果在两个图像中都找到了角点，则保存这些点
        if ret_left and ret_right:
            print("  -> 这对图像可用于校准。")
            objpoints.append(objp)

            # 将角点位置优化到亚像素精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            imgpoints_left.append(corners_left_subpix)
            imgpoints_right.append(corners_right_subpix)
        else:
            print("  -> 这对图像无效，将被跳过。")

    cv2.destroyAllWindows()  # 关闭所有可视化窗口

    # --- 7. 执行校准 ---
    if not objpoints:
        print("\n------------------------------------------------------------")
        print("错误：在任何图像对中都未找到有效的棋盘格角点。")
        print("请检查以下几点：")
        print(
            f"1. `chessboard_size` ({chessboard_size[0]}, {chessboard_size[1]}) 是否正确？这必须是内部角点的数量，而不是方块的数量。")
        print("2. 棋盘格图像是否清晰、光照良好，并且在每张照片中都能看到完整的棋盘格？")
        print("------------------------------------------------------------")
        return None

    print(f"\n使用 {len(objpoints)} 对有效的图像进行校准。")

    # 单独校准每个相机以获得初始内参
    print("正在进行初始的单相机校准...")
    ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_left, gray_shape, None, None)
    ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_right, gray_shape, None, None)

    # 执行立体校准
    print("正在进行立体校准...")
    stereocalib_flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参，因为上面已经确定了
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret_stereo, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K_l, D_l, K_r, D_r, gray_shape,
        criteria=stereocalib_criteria,
        flags=stereocalib_flags
    )

    print("立体校准成功。RMS重投影误差:", ret_stereo)

    # --- 8. 校正并保存参数 ---
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
             map1_right=map1_right, map2_right=map2_right)
    print(f"校准参数已保存至 {save_file_path}")

    return True  # 返回成功标志


if __name__ == '__main__':
    # --- 重要提示：请验证以下参数 ---
    left_calib_img_dir = "../data/calibration_images/left"
    right_calib_img_dir = "../data/calibration_images/right"
    output_params_dir = "params"

    # 这是内部角点的数量。
    # 对于一个有 20x18 个方块的棋盘，你应该使用 (19, 17)。
    # 请再次检查您的实体棋盘格。
    chessboard_corners_cols = 18
    chessboard_corners_rows = 16

    # 一个方块的边长，使用您选择的单位（例如，毫米）。
    square_side_length_mm = 45

    print("开始立体校准流程...")
    calibrate_stereo_camera(
        left_calib_img_dir,
        right_calib_img_dir,
        (chessboard_corners_cols, chessboard_corners_rows),
        square_side_length_mm,
        output_params_dir
    )
    print("流程结束。")