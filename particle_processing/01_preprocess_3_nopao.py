# particle_processing/01_preprocess.py
import cv2
import numpy as np
import glob
import os


def remove_foam_by_shape(image, params):
    """
    通过轮廓的面积和形状（圆度）来识别并去除泡沫。
    """
    # 从参数字典中获取阈值
    min_foam_area = params.get('min_foam_area', 500)
    max_foam_circularity = params.get('max_foam_circularity', 0.5)  # 新增：圆度阈值

    # 使用Otsu's二值化来自动找到一个好的阈值
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 查找所有白色物体的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)

        # (已修改) 增加圆度判断
        # 只有当面积够大，并且形状不圆时，才判定为泡沫
        if area > min_foam_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue  # 避免除以零

            # 计算圆度: 4*pi*Area / (Perimeter^2)。越接近1表示越圆。
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < max_foam_circularity:
                # 将该轮廓区域用黑色填充，即“擦除”泡沫
                cv2.drawContours(output_image, [contour], -1, (0), thickness=cv2.FILLED)

    return output_image


def preprocess_image_pair(img_left_raw, img_right_raw, map1_l, map2_l, map1_r, map2_r, roi_l=None, roi_r=None,
                          perform_cropping=True, clahe_params=None, foam_params=None):
    """
    对图像对进行立体校正和预处理。
    """
    # 步骤 1: 使用remap进行立体校正
    rectified_left = cv2.remap(img_left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right_raw, map1_r, map2_r, cv2.INTER_LINEAR)

    # 步骤 2: (可选) 使用ROI裁剪图像
    if perform_cropping:
        if roi_l is not None:
            x, y, w, h = roi_l
            rectified_left = rectified_left[y:y + h, x:x + w]
        if roi_r is not None:
            x, y, w, h = roi_r
            rectified_right = rectified_right[y:y + h, x:x + w]

    # 步骤 3: 转换为灰度图
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    # 步骤 4: 使用CLAHE
    if clahe_params:
        clahe = cv2.createCLAHE(clipLimit=clahe_params['clipLimit'], tileGridSize=clahe_params['tileGridSize'])
        equalized_left = clahe.apply(gray_left)
        equalized_right = clahe.apply(gray_right)
    else:
        equalized_left = gray_left
        equalized_right = gray_right

    # 步骤 5: 去除泡沫
    if foam_params:
        foam_removed_left = remove_foam_by_shape(equalized_left, foam_params)
        foam_removed_right = remove_foam_by_shape(equalized_right, foam_params)
    else:
        foam_removed_left = equalized_left
        foam_removed_right = equalized_right

    # 步骤 6: 高斯模糊
    blurred_left = cv2.GaussianBlur(foam_removed_left, (5, 5), 0)
    blurred_right = cv2.GaussianBlur(foam_removed_right, (5, 5), 0)

    return blurred_left, blurred_right


def run_preprocessing(calib_params_file, left_image_dir, right_image_dir, output_dir_left, output_dir_right,
                      perform_cropping=True, clahe_params=None, foam_params=None):
    if not os.path.exists(output_dir_left): os.makedirs(output_dir_left)
    if not os.path.exists(output_dir_right): os.makedirs(output_dir_right)

    try:
        print(f"正在从 {calib_params_file} 加载校准数据...")
        calib_data = np.load(calib_params_file)
        map1_l, map2_l = calib_data['map1_left'], calib_data['map2_left']
        map1_r, map2_r = calib_data['map1_right'], calib_data['map2_right']
        roi_l, roi_r = calib_data.get('roi_left'), calib_data.get('roi_right')
        print("校准数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 校准文件 {calib_params_file} 未找到。")
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

        if img_l_raw is None or img_r_raw is None:
            print(f"警告: 无法读取图像对 {os.path.basename(f_left)}")
            continue

        preprocessed_l, preprocessed_r = preprocess_image_pair(img_l_raw, img_r_raw, map1_l, map2_l, map1_r, map2_r,
                                                               roi_l, roi_r, perform_cropping, clahe_params,
                                                               foam_params)

        cv2.imwrite(os.path.join(output_dir_left, f"preprocessed_frame_{i:05d}.png"), preprocessed_l)
        cv2.imwrite(os.path.join(output_dir_right, f"preprocessed_frame_{i:05d}.png"), preprocessed_r)

        if (i + 1) % 100 == 0 or (i + 1) == len(left_image_files):
            print(f"已预处理 {i + 1}/{len(left_image_files)} 对图像。")

    print("所有图像预处理完成。")


if __name__ == '__main__':
    calibration_file = "../camera_calibration/params/stereo_calib_params_from_matlab.npz"

    raw_left_dir = "../data/left_images/"
    raw_right_dir = "../data/right_images/"
    out_preprocessed_left_dir = "../data/preprocessed/left/"
    out_preprocessed_right_dir = "../data/preprocessed/right/"

    # --- 在这里配置处理选项 ---
    # (已修改) 设置为 False 来输出完整的、未裁剪的图像
    CROP_TO_ROI = False

    clahe_config = {
        'clipLimit': 2.0,
        'tileGridSize': (8, 8)
    }

    # (已修改) 智能去泡沫参数配置
    foam_removal_config = {
        'min_foam_area': 500,  # 面积大于此值的物体才被考虑为泡沫
        'max_foam_circularity': 0.5  # 圆度小于此值的物体才被认为是泡沫（越不圆，值越小）
    }

    run_preprocessing(
        calibration_file,
        raw_left_dir,
        raw_right_dir,
        out_preprocessed_left_dir,
        out_preprocessed_right_dir,
        perform_cropping=CROP_TO_ROI,
        clahe_params=clahe_config,
        foam_params=foam_removal_config
    )
