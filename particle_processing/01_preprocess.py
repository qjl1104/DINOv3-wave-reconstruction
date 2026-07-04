# particle_processing/01_preprocess.py
import cv2
import numpy as np
import glob
import os


def preprocess_image_pair(img_left_raw, img_right_raw, map1_l, map2_l, map1_r, map2_r, roi_l=None, roi_r=None,
                          perform_cropping=True, clahe_params=None):
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

    # 步骤 4: 使用CLAHE（对比度受限的自适应直方图均衡化）
    if clahe_params:
        clahe = cv2.createCLAHE(clipLimit=clahe_params['clipLimit'], tileGridSize=clahe_params['tileGridSize'])
        equalized_left = clahe.apply(gray_left)
        equalized_right = clahe.apply(gray_right)
    else:
        equalized_left = gray_left
        equalized_right = gray_right

    # 步骤 5: 对均衡化后的图像进行高斯模糊
    blurred_left = cv2.GaussianBlur(equalized_left, (5, 5), 0)
    blurred_right = cv2.GaussianBlur(equalized_right, (5, 5), 0)

    return blurred_left, blurred_right


def run_preprocessing(calib_params_file, left_image_dir, right_image_dir, output_dir_left, output_dir_right,
                      perform_cropping=True, clahe_params=None):
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

        # (新增) 为第一帧生成校正前后对比图
        if i == 0:
            print("正在为第一帧生成校正前后对比图...")
            # 仅为了对比，我们对原始彩色图像进行校正
            rectified_left_color = cv2.remap(img_l_raw.copy(), map1_l, map2_l, cv2.INTER_LINEAR)

            # 添加文字标签
            cv2.putText(img_l_raw, "Original (Before Rectify)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.putText(rectified_left_color, "Rectified (After Rectify)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 255), 4)

            # 将原图和校正后的图水平拼接
            comparison_image = np.hstack([img_l_raw, rectified_left_color])

            # 保存对比图
            comp_path = os.path.join(os.path.dirname(output_dir_left), "rectify_comparison.jpg")
            cv2.imwrite(comp_path, comparison_image)
            print(f"对比图已保存至: {comp_path}")

        # --- 继续执行常规的预处理流程 ---
        preprocessed_l, preprocessed_r = preprocess_image_pair(img_l_raw, img_r_raw, map1_l, map2_l, map1_r, map2_r,
                                                               roi_l, roi_r, perform_cropping, clahe_params)

        cv2.imwrite(os.path.join(output_dir_left, f"preprocessed_frame_{i:05d}.png"), preprocessed_l)
        cv2.imwrite(os.path.join(output_dir_right, f"preprocessed_frame_{i:05d}.png"), preprocessed_r)

        if (i + 1) % 100 == 0 or (i + 1) == len(left_image_files):
            print(f"已预处理 {i + 1}/{len(left_image_files)} 对图像。")

    print("所有图像预处理完成。")


if __name__ == '__main__':
    # 确保这里的校准文件是您确认质量最好的那一个
    # calibration_file = "../camera_calibration/params/stereo_calib_params_from_matlab_full.npz"
    calibration_file = "../camera_calibration/params/stereo_calib_params_from_matlab.npz"

    raw_left_dir = "../data/left_images/"
    raw_right_dir = "../data/right_images/"
    out_preprocessed_left_dir = "../data/preprocessed/left/"
    out_preprocessed_right_dir = "../data/preprocessed/right/"

    # --- 在这里配置处理选项 ---
    # 您可以切换此项来对比裁剪和不裁剪的效果
    CROP_TO_ROI = False

    clahe_config = {
        'clipLimit': 2.0,
        'tileGridSize': (8, 8)
    }

    run_preprocessing(
        calibration_file,
        raw_left_dir,
        raw_right_dir,
        out_preprocessed_left_dir,
        out_preprocessed_right_dir,
        perform_cropping=CROP_TO_ROI,
        clahe_params=clahe_config
    )
