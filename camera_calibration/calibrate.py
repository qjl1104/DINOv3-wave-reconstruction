# camera_calibration/calibrate.py
import cv2
import numpy as np
import glob
import os

def calibrate_stereo_camera(image_dir_left, image_dir_right, chessboard_size, square_size, params_save_path):
    # 棋盘格角点世界坐标 (Z=0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []       # 存储世界坐标系中的点
    imgpoints_left = []  # 存储左相机图像中的角点
    imgpoints_right = [] # 存储右相机图像中的角点

    images_left = sorted(glob.glob(os.path.join(image_dir_left, '*.bmp'))) # <--- 修改为.bmp
    images_right = sorted(glob.glob(os.path.join(image_dir_right, '*.bmp'))) # <--- 修改为.bmp

    if not images_left or not images_right:
        print(f"Error: Calibration images not found in {image_dir_left} or {image_dir_right} (expected.bmp)")
        return None
    if len(images_left) != len(images_right):
        print("Error: Number of left and right calibration images do not match.")
        return None

    print(f"Found {len(images_left)} image pairs for calibration.")

    # 获取图像尺寸，假设所有标定图像尺寸一致
    img_temp = cv2.imread(images_left[0])
    if img_temp is None:
        print(f"Error: Could not read image {images_left[0]}")
        return None
    gray_shape = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY).shape[::-1] # (width, height)

    for i, (fname_left, fname_right) in enumerate(zip(images_left, images_right)):
        img_left = cv2.imread(fname_left)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.imread(fname_right)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

        if ret_left and ret_right:
            print(f"Chessboard corners found in pair {i+1}: {os.path.basename(fname_left)}, {os.path.basename(fname_right)}")
            objpoints.append(objp)

            # 亚像素角点优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            imgpoints_left.append(corners_left_subpix)
            imgpoints_right.append(corners_right_subpix)
        else:
            print(f"Chessboard not found in {os.path.basename(fname_left)} or {os.path.basename(fname_right)}")

    if not objpoints:
        print("Error: No valid chessboard corners found in any image pair.")
        return None

    print(f"Using {len(objpoints)} valid image pairs for calibration.")

    # 单独标定每个相机获取初始内参
    ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_left, gray_shape, None, None)
    ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_right, gray_shape, None, None)

    # 双目标定
    stereocalib_flags = cv2.CALIB_FIX_INTRINSIC 
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret_stereo, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K_l, D_l, K_r, D_r, gray_shape,
        criteria=stereocalib_criteria,
        flags=stereocalib_flags
    )

    if not ret_stereo: # ret_stereo is the RMS error
        print("Stereo calibration might have high error or failed.")
        # return None # Decide if you want to stop or proceed with potentially bad calibration

    print("Stereo calibration successful. RMS re-projection error:", ret_stereo)

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
    print(f"Calibration parameters saved to {save_file_path}")

    return K_left, D_left, K_right, D_right, R, T, E, F, P1, P2, Q, roi_left, roi_right, map1_left, map2_left, map1_right, map2_right

if __name__ == '__main__':
    left_calib_img_dir = "../data/calibration_images/left"
    right_calib_img_dir = "../data/calibration_images/right"
    output_params_dir = "params"

    chessboard_corners_cols = 19
    chessboard_corners_rows = 17
    square_side_length_mm = 45

    calibrate_stereo_camera(
        left_calib_img_dir,
        right_calib_img_dir,
        (chessboard_corners_cols, chessboard_corners_rows),
        square_side_length_mm,
        output_params_dir
    )