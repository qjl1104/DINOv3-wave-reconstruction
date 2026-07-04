# particle_processing/02_particle_detection.py
import cv2
import numpy as np
import glob
import os
import pickle


def detect_particles_blob(preprocessed_image, detection_params):
    """
    使用OpenCV的SimpleBlobDetector和传入的参数来检测图像中的粒子。
    """
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = detection_params['minArea']
    params.maxArea = detection_params['maxArea']

    params.filterByCircularity = True
    params.minCircularity = detection_params['minCircularity']

    params.filterByConvexity = True
    params.minConvexity = detection_params['minConvexity']

    params.filterByInertia = True
    params.minInertiaRatio = detection_params['minInertiaRatio']

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(preprocessed_image)

    particle_centers = [kp.pt for kp in keypoints]
    return particle_centers


def run_detection(preprocessed_dir_left, preprocessed_dir_right, output_file_left, output_file_right, params_left,
                  params_right):
    """
    遍历所有预处理后的图像，使用不同的参数检测粒子，并将结果保存。
    """
    detections_all_frames_left = []
    detections_all_frames_right = []

    left_files = sorted(glob.glob(os.path.join(preprocessed_dir_left, '*.png')))
    right_files = sorted(glob.glob(os.path.join(preprocessed_dir_right, '*.png')))

    if not left_files or not right_files or len(left_files) != len(right_files):
        print("检测错误: 预处理图像列表 (*.png) 不匹配或为空。")
        return

    print(f"找到 {len(left_files)} 对预处理图像，开始粒子检测...")

    print("\n--- 正在处理左侧相机 ---")
    num_detections_left = 0
    for i, f_left in enumerate(left_files):
        img_l = cv2.imread(f_left, cv2.IMREAD_GRAYSCALE)
        if img_l is None:
            detections_all_frames_left.append([])
            continue
        particles_left = detect_particles_blob(img_l, params_left)
        detections_all_frames_left.append(particles_left)
        num_detections_left += len(particles_left)
        if (i + 1) % 200 == 0 or (i + 1) == len(left_files):
            print(f"  已检测 {i + 1}/{len(left_files)} 帧。")

    print("\n--- 正在处理右侧相机 ---")
    num_detections_right = 0
    for i, f_right in enumerate(right_files):
        img_r = cv2.imread(f_right, cv2.IMREAD_GRAYSCALE)
        if img_r is None:
            detections_all_frames_right.append([])
            continue
        particles_right = detect_particles_blob(img_r, params_right)
        detections_all_frames_right.append(particles_right)
        num_detections_right += len(particles_right)
        if (i + 1) % 200 == 0 or (i + 1) == len(right_files):
            print(f"  已检测 {i + 1}/{len(right_files)} 帧。")

    print("\n--- 检测结果统计 ---")
    print(f"左侧相机总检测点数: {num_detections_left}")
    print(f"右侧相机总检测点数: {num_detections_right}")

    os.makedirs(os.path.dirname(output_file_left), exist_ok=True)
    with open(output_file_left, 'wb') as f:
        pickle.dump(detections_all_frames_left, f)
    print(f"\n左侧相机检测结果已保存至 {output_file_left}")

    os.makedirs(os.path.dirname(output_file_right), exist_ok=True)
    with open(output_file_right, 'wb') as f:
        pickle.dump(detections_all_frames_right, f)
    print(f"右侧相机检测结果已保存至 {output_file_right}")


if __name__ == '__main__':
    prep_left_dir = "../data/preprocessed/left/"
    prep_right_dir = "../data/preprocessed/right/"
    out_det_left_file = "../data/detections/detections_left.pkl"
    out_det_right_file = "../data/detections/detections_right.pkl"

    # --- 参数调试区 ---
    # 我们的目标是：通过调整下面的参数，让最终输出的“左侧相机总检测点数”
    # 和“右侧相机总检测点数”大致相当。

    # --- 左侧相机参数 (需要仔细调整) ---
    # 由于左侧相机之前产生了大量噪声，我们从一套非常严格的参数开始。
    # 如果检测点太少，可以逐步“放宽”这些值。
    detection_params_left = {
        'minArea': 25,  # 如果检测点太少，可以适当减小这个值 (例如 22, 20, 18...)
        'maxArea': 300,
        'minCircularity': 0.75,  # 如果检测点太少，可以适当减小这个值 (例如 0.7, 0.65...)
        'minConvexity': 0.87,  # 如果检测点太少，可以适当减小这个值 (例如 0.85, 0.82...)
        'minInertiaRatio': 0.4
    }

    # --- 右侧相机参数 (保持不变，作为基准) ---
    # 右侧相机的数据质量较好，我们使用一套标准的参数。
    detection_params_right = {
        'minArea': 25,
        'maxArea': 300,
        'minCircularity': 0.75,
        'minConvexity': 0.87,
        'minInertiaRatio': 0.4
    }

    run_detection(
        prep_left_dir, prep_right_dir,
        out_det_left_file, out_det_right_file,
        detection_params_left, detection_params_right
    )
