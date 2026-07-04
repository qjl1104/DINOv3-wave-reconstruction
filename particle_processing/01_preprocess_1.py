# particle_processing/01_preprocess.py
import cv2
import numpy as np
import glob
import os


def preprocess_image_pair(img_left_raw, img_right_raw, map1_l, map2_l, map1_r, map2_r, roi_l=None, roi_r=None):
    """
    对图像对进行立体校正和预处理。
    此版本会使用ROI来裁剪掉校正后产生的黑边。
    """
    # 步骤 1: 使用remap进行立体校正
    rectified_left = cv2.remap(img_left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right_raw, map1_r, map2_r, cv2.INTER_LINEAR)

    # 步骤 2: (已启用) 使用ROI裁剪图像，移除黑边
    if roi_l is not None:
        x, y, w, h = roi_l
        rectified_left = rectified_left[y:y + h, x:x + w]
    if roi_r is not None:
        x, y, w, h = roi_r
        rectified_right = rectified_right[y:y + h, x:x + w]

    # 步骤 3: 转换为灰度图
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    # 步骤 4: 高斯模糊以平滑图像，减少噪声
    blurred_left = cv2.GaussianBlur(gray_left, (5, 5), 0)
    blurred_right = cv2.GaussianBlur(gray_right, (5, 5), 0)

    return blurred_left, blurred_right


def run_preprocessing(calib_params_file, left_image_dir, right_image_dir, output_dir_left, output_dir_right):
    # 创建输出文件夹
    if not os.path.exists(output_dir_left): os.makedirs(output_dir_left)
    if not os.path.exists(output_dir_right): os.makedirs(output_dir_right)

    # 加载校准参数
    try:
        print(f"正在从 {calib_params_file} 加载校准数据...")
        calib_data = np.load(calib_params_file)
        map1_l, map2_l = calib_data['map1_left'], calib_data['map2_left']
        map1_r, map2_r = calib_data['map1_right'], calib_data['map2_right']
        # (已启用) 加载ROI，使用.get()以防旧的校准文件没有存ROI
        roi_l = calib_data.get('roi_left')
        roi_r = calib_data.get('roi_right')
        print("校准数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 校准文件 {calib_params_file} 未找到。")
        return
    except KeyError as e:
        print(f"错误: 校准文件 {calib_params_file} 中缺少关键参数 {e}。")
        return

    # 获取图像文件列表
    left_image_files = sorted(glob.glob(os.path.join(left_image_dir, '*.bmp')))
    right_image_files = sorted(glob.glob(os.path.join(right_image_dir, '*.bmp')))

    if not left_image_files or not right_image_files:
        print(f"错误: 在 {left_image_dir} 或 {right_image_dir} 中未找到图像 (需要.bmp格式)。")
        return
    if len(left_image_files) != len(right_image_files):
        print("错误: 左右相机图像数量不匹配。")
        return

    print(f"找到 {len(left_image_files)} 对图像，开始预处理...")
    for i, (f_left, f_right) in enumerate(zip(left_image_files, right_image_files)):
        img_l_raw = cv2.imread(f_left)
        img_r_raw = cv2.imread(f_right)

        if img_l_raw is None or img_r_raw is None:
            print(f"警告: 无法读取图像对 {os.path.basename(f_left)}, {os.path.basename(f_right)}")
            continue

        # (已启用) 将ROI参数传递给处理函数
        preprocessed_l, preprocessed_r = preprocess_image_pair(img_l_raw, img_r_raw, map1_l, map2_l, map1_r, map2_r,
                                                               roi_l, roi_r)

        # 保存预处理后的图像为PNG格式
        cv2.imwrite(os.path.join(output_dir_left, f"preprocessed_frame_{i:05d}.png"), preprocessed_l)
        cv2.imwrite(os.path.join(output_dir_right, f"preprocessed_frame_{i:05d}.png"), preprocessed_r)

        if (i + 1) % 50 == 0 or (i + 1) == len(left_image_files):
            print(f"已预处理 {i + 1}/{len(left_image_files)} 对图像。")

    print("所有图像预处理完成。")


if __name__ == '__main__':
    calibration_file = "../camera_calibration/params/stereo_calib_params.npz"
    raw_left_dir = "../data/left_images/"
    raw_right_dir = "../data/right_images/"
    out_preprocessed_left_dir = "../data/preprocessed/left/"
    out_preprocessed_right_dir = "../data/preprocessed/right/"

    run_preprocessing(calibration_file, raw_left_dir, raw_right_dir, out_preprocessed_left_dir,
                      out_preprocessed_right_dir)
