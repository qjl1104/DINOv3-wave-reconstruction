# particle_processing/01_preprocess.py
import cv2
import numpy as np
import glob
import os


def remove_foam_by_shape(image, params):
    """
    通过轮廓的面积、形状（圆度）和亮度来识别并去除泡沫和微小噪点。
    此版本使用固定的二值化阈值以提高帧间稳定性。
    """
    # 从参数字典中获取阈值
    min_particle_area = params.get('min_particle_area', 10)
    min_foam_area = params.get('min_foam_area', 500)
    max_foam_circularity = params.get('max_foam_circularity', 0.5)
    max_foam_intensity = params.get('max_foam_intensity', 150)
    opening_kernel_size = params.get('opening_kernel_size', 3)
    fixed_threshold = params.get('fixed_threshold', 40)  # 新增：固定的二值化阈值

    # (已修改) 使用固定的阈值进行二值化，而不是Otsu's自动阈值
    _, thresh = cv2.threshold(image, fixed_threshold, 255, cv2.THRESH_BINARY)

    # 形态学开运算，断开粒子间的微小连接
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    opened_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 在经过“开运算”处理的图像上查找轮廓
    contours, _ = cv2.findContours(opened_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)

        # 1. 去除面积过小的噪点
        if area < min_particle_area:
            cv2.drawContours(output_image, [contour], -1, (0), thickness=cv2.FILLED)
            continue

        # 2. 去除大块、不圆且不够亮的泡沫
        if area > min_foam_area:
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            mean_val = cv2.mean(image, mask=mask)[0]

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < max_foam_circularity and mean_val < max_foam_intensity:
                cv2.drawContours(output_image, [contour], -1, (0), thickness=cv2.FILLED)

    return output_image


def preprocess_image_pair(img_l_raw, img_r_raw, bg_l_raw, bg_r_raw, map1_l, map2_l, map1_r, map2_r,
                          perform_cropping=True, clahe_params=None, foam_params=None):
    """
    对图像对进行立体校正和预处理，包含背景减除和去泡沫步骤。
    """
    # 步骤 1: 对当前帧和背景模型都进行立体校正
    rectified_left = cv2.remap(img_l_raw, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_r_raw, map1_r, map2_r, cv2.INTER_LINEAR)
    bg_left_rect = cv2.remap(bg_l_raw, map1_l, map2_l, cv2.INTER_LINEAR)
    bg_right_rect = cv2.remap(bg_r_raw, map1_r, map2_r, cv2.INTER_LINEAR)

    # 步骤 2: 转换为灰度图
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    # 步骤 3: 背景减除
    subtracted_left = cv2.subtract(gray_left, bg_left_rect)
    subtracted_right = cv2.subtract(gray_right, bg_right_rect)

    # 步骤 4: 使用CLAHE增强对比度
    if clahe_params:
        clahe = cv2.createCLAHE(clipLimit=clahe_params['clipLimit'], tileGridSize=clahe_params['tileGridSize'])
        enhanced_left = clahe.apply(subtracted_left)
        enhanced_right = clahe.apply(subtracted_right)
    else:
        enhanced_left = subtracted_left
        enhanced_right = subtracted_right

    # 步骤 5: 去除泡沫和噪点
    if foam_params:
        foam_removed_left = remove_foam_by_shape(enhanced_left, foam_params)
        foam_removed_right = remove_foam_by_shape(enhanced_right, foam_params)
    else:
        foam_removed_left = enhanced_left
        foam_removed_right = enhanced_right

    # 步骤 6: 高斯模糊以平滑图像
    blurred_left = cv2.GaussianBlur(foam_removed_left, (5, 5), 0)
    blurred_right = cv2.GaussianBlur(foam_removed_right, (5, 5), 0)

    # 步骤 7: (可选) 裁剪到有效的ROI
    if perform_cropping:
        pass

    return blurred_left, blurred_right


def run_preprocessing(calib_params_file, left_image_dir, right_image_dir, bg_file_left, bg_file_right, output_dir_left,
                      output_dir_right, perform_cropping=True, clahe_params=None, foam_params=None):
    if not os.path.exists(output_dir_left): os.makedirs(output_dir_left)
    if not os.path.exists(output_dir_right): os.makedirs(output_dir_right)

    try:
        print(f"正在从 {calib_params_file} 加载校准数据...")
        calib_data = np.load(calib_params_file)
        map1_l, map2_l = calib_data['map1_left'], calib_data['map2_left']
        map1_r, map2_r = calib_data['map1_right'], calib_data['map2_right']

        print("正在加载背景模型...")
        bg_l = cv2.imread(bg_file_left, cv2.IMREAD_GRAYSCALE)
        bg_r = cv2.imread(bg_file_right, cv2.IMREAD_GRAYSCALE)
        if bg_l is None or bg_r is None:
            print("错误: 未能加载背景模型文件。请先运行 00_generate_background.py。")
            return

    except FileNotFoundError:
        print(f"错误: 校准文件或背景文件未找到。")
        return

    left_image_files = sorted(glob.glob(os.path.join(left_image_dir, '*.bmp')))
    right_image_files = sorted(glob.glob(os.path.join(right_image_dir, '*.bmp')))
    if not left_image_files or not right_image_files:
        print(f"错误: 在 {left_image_dir} 或 {right_image_dir} 中未找到图像。")
        return

    print(f"找到 {len(left_image_files)} 对图像，开始预处理...")
    for i, (f_left, f_right) in enumerate(zip(left_image_files, right_image_files)):
        img_l_raw = cv2.imread(f_left)
        img_r_raw = cv2.imread(f_right)
        if img_l_raw is None or img_r_raw is None: continue

        preprocessed_l, preprocessed_r = preprocess_image_pair(img_l_raw, img_r_raw, bg_l, bg_r, map1_l, map2_l, map1_r,
                                                               map2_r,
                                                               perform_cropping, clahe_params, foam_params)

        cv2.imwrite(os.path.join(output_dir_left, f"preprocessed_frame_{i:05d}.png"), preprocessed_l)
        cv2.imwrite(os.path.join(output_dir_right, f"preprocessed_frame_{i:05d}.png"), preprocessed_r)

        if (i + 1) % 100 == 0 or (i + 1) == len(left_image_files):
            print(f"已预处理 {i + 1}/{len(left_image_files)} 对图像。")

    print("所有图像预处理完成。")


if __name__ == '__main__':
    calibration_file = "../camera_calibration/params/stereo_calib_params_from_matlab_full.npz"

    raw_left_dir = "../data/left_images/"
    raw_right_dir = "../data/right_images/"

    bg_left_path = "../data/preprocessed/background_left.png"
    bg_right_path = "../data/preprocessed/background_right.png"

    out_preprocessed_left_dir = "../data/preprocessed/left/"
    out_preprocessed_right_dir = "../data/preprocessed/right/"

    # --- 在这里配置处理选项 ---
    CROP_TO_ROI = False

    clahe_config = {
        'clipLimit': 4.0,
        'tileGridSize': (8, 8)
    }

    # (已修改) 智能去泡沫/噪点参数配置
    foam_removal_config = {
        'fixed_threshold': 40,  # 新增：固定的亮度阈值 (0-255)，值越低，能识别越暗的点
        'min_particle_area': 15,
        'min_foam_area': 500,
        'max_foam_circularity': 0.5,
        'max_foam_intensity': 150,
        'opening_kernel_size': 3
    }

    run_preprocessing(
        calibration_file,
        raw_left_dir,
        raw_right_dir,
        bg_left_path,
        bg_right_path,
        out_preprocessed_left_dir,
        out_preprocessed_right_dir,
        perform_cropping=CROP_TO_ROI,
        clahe_params=clahe_config,
        foam_params=foam_removal_config
    )
