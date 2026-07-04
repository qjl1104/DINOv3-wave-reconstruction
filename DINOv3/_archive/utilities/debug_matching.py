# debug_matching.py
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


# --- 直接从您的代码中复制关键函数 ---
# 我们可以直接导入，但为了独立性，这里复制过来
def detect_markers(image_path: str) -> np.ndarray:
    """检测图像中的标记点"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return np.array([])
    # 阈值可以根据实际情况微调
    _, binary_img = cv2.threshold(img, 68, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 8 < area < 500 and len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                hull = cv2.convexHull(contour)
                solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                (w, h) = ellipse[1]
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100
                if solidity > 0.85 and aspect_ratio < 8.0:
                    marker_centers.append(ellipse[0])
            except cv2.error:
                continue
    return np.array(marker_centers, dtype=np.float32)


# --- 可视化函数 ---
def visualize_marker_matching(l_path, r_path):
    """
    可视化左右图像的标记点检测和潜在匹配，帮助调试。
    """
    print(f"正在处理左图: {l_path}")
    print(f"正在处理右图: {r_path}")

    left_img = cv2.imread(l_path)
    right_img = cv2.imread(r_path)

    if left_img is None or right_img is None:
        print("错误：无法读取一张或两张图片。")
        return

    left_markers = detect_markers(l_path)
    right_markers = detect_markers(r_path)

    print(f"在左图中检测到 {len(left_markers)} 个标记点。")
    print(f"在右图中检测到 {len(right_markers)} 个标记点。")

    if len(left_markers) == 0 or len(right_markers) == 0:
        print("其中一张图像未检测到标记点，无法继续。")
        return

    # 创建一个大的画布，将两张图并排放置
    h, w, _ = left_img.shape
    combined_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined_img[:, :w] = left_img
    combined_img[:, w:] = right_img

    # 在合并后的图像上画出所有检测到的点
    # 左图点为绿色
    for x, y in left_markers:
        cv2.circle(combined_img, (int(x), int(y)), 5, (0, 255, 0), -1)
    # 右图点为蓝色
    for x, y in right_markers:
        cv2.circle(combined_img, (int(x) + w, int(y)), 5, (255, 0, 0), -1)

    # --- 尝试匹配并画线 ---
    # 我们放宽条件来观察潜在的匹配
    potential_matches = 0
    correct_condition_matches = 0
    for i, p_l in enumerate(left_markers):
        for j, p_r in enumerate(right_markers):
            # 放宽y的限制，便于观察
            if abs(p_l[1] - p_r[1]) < 20:
                p1_x, p1_y = int(p_l[0]), int(p_l[1])
                p2_x, p2_y = int(p_r[0]) + w, int(p_r[1])

                # 检查原始脚本中的条件是否满足
                # 如果满足，画红色实线
                if abs(p_l[1] - p_r[1]) < 10 and p_l[0] > p_r[0]:
                    cv2.line(combined_img, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
                    correct_condition_matches += 1
                # 如果不满足，画黄色虚线，帮助我们看到“本应匹配但失败”的点
                else:
                    cv2.line(combined_img, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 255), 1)
                potential_matches += 1

    print(f"找到了 {potential_matches} 对y坐标相近的潜在匹配。")
    print(f"其中只有 {correct_condition_matches} 对满足原始脚本的严格条件。")

    # 显示图像
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.title("标记点匹配可视化 (左:绿点, 右:蓝点)\n红线:满足条件; 黄线:潜在匹配")
    plt.axis('off')

    # 保存结果图像以便仔细分析
    output_filename = "debug_matching_visualization.png"
    plt.savefig(output_filename)
    print(f"\n[✓] 结果已保存为: {output_filename}")
    plt.show()


if __name__ == '__main__':
    # --- 配置您的路径 ---
    # !!! 请确保这里的路径与您主脚本中的Config类一致 !!!
    LEFT_IMAGE_DIR = "D:/Research/wave_reconstruction_project/data/lresult/"
    RIGHT_IMAGE_DIR = "D:/Research/wave_reconstruction_project/data/rresult/"

    # 寻找第一张图片进行调试
    left_image_paths = sorted(glob.glob(os.path.join(LEFT_IMAGE_DIR, "*.*")))
    if not left_image_paths:
        print(f"错误：在 '{LEFT_IMAGE_DIR}' 中找不到任何图像文件。请检查路径。")
    else:
        l_path_to_debug = left_image_paths[0]  # 默认调试第一帧
        basename = os.path.basename(l_path_to_debug)
        r_basename = basename.replace('lresult', 'rresult')
        r_path_to_debug = os.path.join(RIGHT_IMAGE_DIR, r_basename)

        if not os.path.exists(r_path_to_debug):
            print(f"错误：找不到对应的右图 '{r_path_to_debug}'")
        else:
            visualize_marker_matching(l_path_to_debug, r_path_to_debug)