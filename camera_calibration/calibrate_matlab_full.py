# use_matlab_params_full_view.py
import cv2
import numpy as np
import os


def generate_maps_from_matlab_params(image_for_size_left, image_for_size_right, params_save_path):
    """
    根据MATLAB标定得到的内外参数，生成并保存立体校正所需的文件。
    此版本使用 alpha=1 模式，以保留所有原始图像像素。
    """
    print("正在从MATLAB参数生成立体校正映射...")

    # --- 1. 定义相机参数 ---
    fx_l, fy_l = 3937.20908, 3936.40700;
    cx_l, cy_l = 1349.85569, 952.73155
    K_left = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]], dtype=np.float64)
    fx_r, fy_r = 3925.72662, 3924.51774;
    cx_r, cy_r = 1278.88564, 886.81188
    K_right = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]], dtype=np.float64)
    D_left = np.array([-2.593e-2, 32.052e-2, 1.046e-2, 0.321e-2, 0], dtype=np.float64)
    D_right = np.array([-2.287e-2, 47.839e-2, 0.656e-2, -0.27e-2, 0], dtype=np.float64)
    rvec = np.array([-0.03702, 0.25002, 0.1387], dtype=np.float64)
    T = np.array([-1397.67526, -141.94746, 153.43894], dtype=np.float64)

    # --- 2. 获取图像尺寸 ---
    img_left = cv2.imread(image_for_size_left)
    if img_left is None:
        print(f"错误：无法读取图像文件 '{image_for_size_left}'。")
        return

    image_size = (img_left.shape[1], img_left.shape[0])
    print(f"获取到的图像尺寸 (宽x高): {image_size}")

    # --- 3. 计算校正参数 ---
    R, _ = cv2.Rodrigues(rvec)
    tx, ty, tz = T[0], T[1], T[2]
    T_skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    E = T_skew @ R
    F = np.linalg.inv(K_right).T @ E @ np.linalg.inv(K_left)

    # (已修改) --- 使用 alpha=1 来保留所有像素 ---
    # 这将确保远处的点不会被裁剪掉。
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, image_size, R, T,
        alpha=1  # <--- 关键修改在这里！
    )
    print("立体校正计算完成 (使用 alpha=1 全景模式)。")

    # --- 4. 生成查找映射 ---
    map1_left, map2_left = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_16SC2)
    print("无畸变校正映射生成完毕。")

    # --- 5. 保存所有参数到 .npz 文件 ---
    save_dir = os.path.dirname(params_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(params_save_path,
             K_left=K_left, D_left=D_left, K_right=K_right, D_right=D_right,
             R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             image_size=image_size, map1_left=map1_left, map2_left=map2_left,
             map1_right=map1_right, map2_right=map2_right,
             roi_left=roi_left, roi_right=roi_right)
    print(f"所有参数和映射已成功保存到: {params_save_path}")


if __name__ == '__main__':
    # 请确保这里的路径是正确的
    left_image_path = "D:/Research/wave_reconstruction_project/data/left_images/left0001.bmp"
    right_image_path = "D:/Research/wave_reconstruction_project/data/right_images/right0001.bmp"

    # 定义最终参数文件的保存位置
    output_params_file = "D:/Research/wave_reconstruction_project/camera_calibration/params/stereo_calib_params_from_matlab_full.npz"

    if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
        print(f"错误: 无法找到图像文件，请检查路径。")
    else:
        generate_maps_from_matlab_params(
            left_image_path,
            right_image_path,
            output_params_file
        )
