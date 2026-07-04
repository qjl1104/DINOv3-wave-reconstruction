# particle_processing/01_preprocess.py
import cv2
import numpy as np
import glob
import os

from sqlalchemy.sql.operators import truediv


def preprocess_image_pair(img_left_raw, img_right_raw, map1_l, map2_l, map1_r, map2_r, roi_l=None, roi_r=None,
                          roi_scale_factor=1.0, perform_cropping=True):
    """
    对图像对进行立体校正和预处理。
    """
    # 步骤 1: 使用remap进行立体校正
    rectified_left = cv2.remap(img_left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right_raw, map1_r, map2_r, cv2.INTER_LINEAR)

    # 步骤 2: (可选择) 根据开关决定是否使用ROI裁剪图像
    if perform_cropping:
        if roi_l is not None:
            rectified_left = scale_and_crop_roi(rectified_left, roi_l, roi_scale_factor)
        if roi_r is not None:
            rectified_right = scale_and_crop_roi(rectified_right, roi_r, roi_scale_factor)

    # 步骤 3: 转换为灰度图
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    # 步骤 4: 高斯模糊以平滑图像，减少噪声
    blurred_left = cv2.GaussianBlur(gray_left, (5, 5), 0)
    blurred_right = cv2.GaussianBlur(gray_right, (5, 5), 0)

    return blurred_left, blurred_right


def scale_and_crop_roi(image, roi, scale_factor):
    """
    根据缩放因子调整ROI并裁剪图像。
    """
    x, y, w, h = roi
    img_h, img_w = image.shape[:2]
    center_x, center_y = x + w / 2, y + h / 2
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    new_x, new_y = int(center_x - new_w / 2), int(center_y - new_h / 2)
    new_x, new_y = max(0, new_x), max(0, new_y)
    new_w, new_h = min(img_w - new_x, new_w), min(img_h - new_y, new_h)
    return image[new_y:new_y + new_h, new_x:new_x + new_w]


def run_preprocessing(calib_params_file, left_image_dir, right_image_dir, output_dir_left, output_dir_right,
                      roi_scale_factor=1.0, perform_cropping=True, downscale_if_mismatch=False):
    if not os.path.exists(output_dir_left): os.makedirs(output_dir_left)
    if not os.path.exists(output_dir_right): os.makedirs(output_dir_right)

    try:
        print(f"正在从 {calib_params_file} 加载校准数据...")
        calib_data = np.load(calib_params_file)
        map1_l, map2_l = calib_data['map1_left'], calib_data['map2_left']
        map1_r, map2_r = calib_data['map1_right'], calib_data['map2_right']
        roi_l, roi_r = calib_data.get('roi_left'), calib_data.get('roi_right')
        calib_image_size = tuple(calib_data['image_size'])
        print(f"校准数据加载成功。校准时使用的图像尺寸为: {calib_image_size}")
    except FileNotFoundError:
        print(f"错误: 校准文件 {calib_params_file} 未找到。")
        return
    except KeyError as e:
        print(f"错误: 校准文件 {calib_params_file} 中缺少关键参数 {e}。")
        return

    left_image_files = sorted(glob.glob(os.path.join(left_image_dir, '*.bmp')))
    right_image_files = sorted(glob.glob(os.path.join(right_image_dir, '*.bmp')))

    if not left_image_files or not right_image_files:
        print(f"错误: 在 {left_image_dir} 或 {right_image_dir} 中未找到图像 (需要.bmp格式)。")
        return
    if len(left_image_files) != len(right_image_files):
        print("错误: 左右相机图像数量不匹配。")
        return

    first_img_raw = cv2.imread(left_image_files[0])
    if first_img_raw is None:
        print(f"错误：无法读取第一张待处理图像 {left_image_files[0]}。")
        return

    processing_img_h, processing_img_w = first_img_raw.shape[:2]
    processing_img_size = (processing_img_w, processing_img_h)

    if calib_image_size != processing_img_size:
        if downscale_if_mismatch:
            print("\n警告: 图像尺寸不匹配！")
            print(f"  > 校准尺寸: {calib_image_size}, 图像尺寸: {processing_img_size}")
            print(f"  > 已启用自动缩放，将把图像从 {processing_img_size} 缩小到 {calib_image_size} 进行处理。\n")
        else:
            print("\n" + "=" * 60)
            print("致命错误：图像尺寸不匹配！")
            print(f"  > 校准文件中记录的尺寸 (宽, 高): {calib_image_size}")
            print(f"  > 当前待处理图像的尺寸 (宽, 高): {processing_img_size}")
            print("  > 解决方法1 (推荐): 使用 {processing_img_size} 分辨率的图像重新校准。")
            print(f"  > 解决方法2 (权宜之计): 在主函数中设置 `DOWNSCALE_TO_MATCH_CALIBRATION = True`。")
            print("=" * 60 + "\n")
            return
    else:
        print("图像尺寸匹配检查通过。")

    print(f"找到 {len(left_image_files)} 对图像，开始预处理...")
    for i, (f_left, f_right) in enumerate(zip(left_image_files, right_image_files)):
        img_l_raw = cv2.imread(f_left)
        img_r_raw = cv2.imread(f_right)

        if img_l_raw is None or img_r_raw is None:
            print(f"警告: 无法读取图像对 {os.path.basename(f_left)}, {os.path.basename(f_right)}")
            continue

        # (新增) 如果尺寸不匹配且已启用缩放，则执行缩放
        if calib_image_size != processing_img_size and downscale_if_mismatch:
            img_l_raw = cv2.resize(img_l_raw, calib_image_size, interpolation=cv2.INTER_AREA)
            img_r_raw = cv2.resize(img_r_raw, calib_image_size, interpolation=cv2.INTER_AREA)

        preprocessed_l, preprocessed_r = preprocess_image_pair(img_l_raw, img_r_raw, map1_l, map2_l, map1_r, map2_r,
                                                               roi_l, roi_r, roi_scale_factor, perform_cropping)

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

    # --- 在这里配置处理选项 ---
    # 步骤 1: 如果待处理图像与校准图像尺寸不匹配，是否自动缩小待处理图像？
    # False (推荐) = 打印错误并停止，提醒您重新校准。
    # True (权宜之计) = 将大图缩小以匹配校准尺寸，会损失精度。
    DOWNSCALE_TO_MATCH_CALIBRATION = True

    # 步骤 2: 是否裁剪校正后的图像以移除黑边？
    CROP_TO_ROI = True

    # 步骤 3: 如果 CROP_TO_ROI 为 True，可调整此因子放大视野
    scaling_factor = 1.0

    run_preprocessing(
        calibration_file,
        raw_left_dir,
        raw_right_dir,
        out_preprocessed_left_dir,
        out_preprocessed_right_dir,
        roi_scale_factor=scaling_factor,
        perform_cropping=CROP_TO_ROI,
        downscale_if_mismatch=DOWNSCALE_TO_MATCH_CALIBRATION
    )
