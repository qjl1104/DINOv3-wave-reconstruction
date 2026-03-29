# verify_calibration.py
# 描述:
# 这是一个诊断工具，用于帮助你验证你现有的标定文件
# ('stereo_calib_params_from_matlab_full.npz') 中的 P1 和 P2 投影矩阵
# 是否真的适用于校正后的图像。
# V3.0: 修复了文件路径问题，使其能够自动定位标定文件。

import cv2
import numpy as np
import os

# --- 全局变量用于存储点击的点 ---
points_left = []
points_right = []


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于记录点击坐标"""
    if event == cv2.EVENT_LBUTTONDOWN:
        img_display = param['image'].copy()
        point_list = param['points']

        point_list.append((x, y))

        # 在图像上绘制已点击的点
        for i, pt in enumerate(point_list):
            cv2.circle(img_display, pt, 5, (0, 255, 0), -1)
            cv2.putText(img_display, str(i + 1), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow(param['window_name'], img_display)
        print(f"在 {param['window_name']} 上记录了点 #{len(point_list)}: ({x}, {y})")


def get_manual_points(image, window_name, num_points):
    """
    显示图像并让用户手动点击指定数量的点。
    """
    points = []
    param = {'image': image, 'points': points, 'window_name': window_name}

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, mouse_callback, param)

    print(f"\n请在窗口 '{window_name}' 中，按顺序精确点击 {num_points} 个特征点。")
    print("完成后，按 'Enter' 键继续...")

    while len(points) < num_points:
        key = cv2.waitKey(0)
        if key == 13:  # Enter key
            if len(points) < num_points:
                print(f"错误: 你只点击了 {len(points)} 个点，需要 {num_points} 个。请继续点击。")
            else:
                break
        elif key == 27:  # ESC key to quit
            cv2.destroyAllWindows()
            return None

    cv2.destroyWindow(window_name)
    return points


def verify_calibration_parameters():
    """主验证函数"""
    # --- 用户配置 ---
    # --- 修改：采用与你原始代码相同的逻辑来构建文件路径 ---
    # 假设此脚本位于 DINOv3 文件夹下
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.dirname(PROJECT_ROOT)
    ORIGINAL_CALIB_FILE = os.path.join(DATA_ROOT, "camera_calibration", "params",
                                       "stereo_calib_params_from_matlab_full.npz")
    # --- 修改结束 ---

    LEFT_IMAGE_PATH = r'D:\Research\wave_reconstruction_project\data\left_images\left0100.bmp'  # 选择一张清晰的左图
    RIGHT_IMAGE_PATH = r'D:\Research\wave_reconstruction_project\data\right_images\right0100.bmp'  # 对应的右图
    NUM_POINTS_TO_VERIFY = 5  # 验证点的数量
    # --- 配置结束 ---

    print("=" * 60)
    print("          相机标定参数 (P1, P2) 验证工具")
    print("=" * 60)

    # 1. 加载标定文件
    if not os.path.exists(ORIGINAL_CALIB_FILE):
        print(f"错误: 原始标定文件 '{ORIGINAL_CALIB_FILE}' 未找到。")
        print("请确保此脚本位于 'DINOv3' 文件夹下。")
        return
    try:
        calib = np.load(ORIGINAL_CALIB_FILE)
        # 检查 P1, P2 是否存在
        if 'P1' not in calib or 'P2' not in calib:
            print(f"错误: 标定文件 '{ORIGINAL_CALIB_FILE}' 中不包含 'P1' 或 'P2' 矩阵。")
            return
        P1, P2 = calib['P1'], calib['P2']
        map1_l, map2_l = calib['map1_left'], calib['map2_left']
        map1_r, map2_r = calib['map1_right'], calib['map2_right']
    except Exception as e:
        print(f"加载标定文件时出错: {e}")
        return

    # 2. 加载并校正图像
    img_l = cv2.imread(LEFT_IMAGE_PATH, 0)
    img_r = cv2.imread(RIGHT_IMAGE_PATH, 0)
    if img_l is None or img_r is None:
        print(f"错误: 无法读取图像。请检查路径:\n{LEFT_IMAGE_PATH}\n{RIGHT_IMAGE_PATH}")
        return

    img_l_rect = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
    img_r_rect = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

    # 3. 手动获取匹配点
    points_l = get_manual_points(img_l_rect, "Left Rectified Image", NUM_POINTS_TO_VERIFY)
    if points_l is None: return

    points_r = get_manual_points(img_r_rect, "Right Rectified Image", NUM_POINTS_TO_VERIFY)
    if points_r is None: return

    if len(points_l) != len(points_r):
        print("错误：左右图像上点击的点数量不匹配。正在中止。")
        return

    pts_l_np = np.array(points_l, dtype=np.float32)
    pts_r_np = np.array(points_r, dtype=np.float32)

    # 4. 计算重投影误差
    print("\n正在使用你文件中的 P1 和 P2 计算重投影误差...")

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts_l_np.T, pts_r_np.T)
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T

    reprojected_l = (P1 @ points_4d_hom)[:2] / (P1 @ points_4d_hom)[2]
    reprojected_r = (P2 @ points_4d_hom)[:2] / (P2 @ points_4d_hom)[2]

    errors_l = np.sqrt(np.sum((pts_l_np - reprojected_l.T) ** 2, axis=1))
    errors_r = np.sqrt(np.sum((pts_r_np - reprojected_r.T) ** 2, axis=1))
    total_errors = errors_l + errors_r

    # 5. 显示结果并给出诊断
    print("\n" + "=" * 60)
    print("                   诊断结果")
    print("=" * 60)
    print("手动选择的匹配点的重投影误差 (单位: 像素):")
    for i, err in enumerate(total_errors):
        print(f"  - 点 #{i + 1}: {err:.4f} 像素")

    avg_error = np.mean(total_errors)
    print(f"\n平均误差: {avg_error:.4f} 像素")

    print("\n--- 结论 ---")
    if avg_error > 5.0 or not np.isfinite(avg_error):
        print("❌ 诊断: 你的 P1 和 P2 矩阵与校正后的图像不匹配。")
        print("   这是一个严重问题，导致 analyze_my_model.py 的定量结果无效。")
        print("\n--- 解决方案 ---")
        print("   请立即运行 'create_rectified_calib.py' 脚本。")
        print("   它会自动为你计算并生成一个包含正确 P1 和 P2 矩阵的全新标定文件。")
    else:
        print("✅ 诊断: 你的 P1 和 P2 矩阵看起来是正确的。")
        print("   如果 analyze_my_model.py 仍然输出0误差，问题可能出在其他地方。")
        print("   但标定文件不匹配的可能性已经基本排除。")
    print("=" * 60)


if __name__ == "__main__":
    verify_calibration_parameters()

