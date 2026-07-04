# visual_debugger.py
# 描述:
# 这是一个终极诊断工具，用于可视化地证明相机标定参数是否存在问题。
# 它会运行你的模型得到匹配点（绿色圆圈），然后使用你的 P1/P2 矩阵
# 进行三维重建和重投影，并将结果画在图上（红色十字）。
#
# 结果解读:
# - ✅ 如果红十字精确地落在绿圈中央：说明你的标定参数是完美的。
# - ❌ 如果红十字远离绿圈，散布在各处：说明你的 P1/P2 矩阵是错误的。

import os
import sys
import cv2
import numpy as np
import torch

try:
    from sparse_reconstructor_1013_gemini import Config, SparseMatchingStereoModel
except ImportError:
    sys.exit("错误: 无法导入 'sparse_reconstructor_1013_gemini.py'。")

# 导入最终版的 ModelAnalyzer，因为它包含了正确的坐标变换逻辑
from analyze_my_model import ModelAnalyzer, find_latest_run_dir


def visualize_reprojection_error():
    """主可视化函数"""
    # --- 用户配置 ---
    IMAGE_INDEX_TO_DEBUG = 100  # 选择一帧用于调试
    # --- 配置结束 ---

    print("=" * 60)
    print("        重投影误差 可视化诊断工具")
    print("=" * 60)

    # 1. 初始化分析器 (它会加载模型和标定文件)
    latest_run_dir = find_latest_run_dir()
    if not latest_run_dir: return

    try:
        analyzer = ModelAnalyzer(run_dir=latest_run_dir)
    except Exception as e:
        print(f"初始化分析器时出错: {e}")
        return

    # 2. 获取匹配点 (使用 analyzer 中已经修复好的函数)
    print(f"\n正在处理图像索引: {IMAGE_INDEX_TO_DEBUG}")
    left_path = analyzer.left_images[IMAGE_INDEX_TO_DEBUG]
    right_path = os.path.join(analyzer.cfg.RIGHT_IMAGE_DIR, 'right' + os.path.basename(left_path)[4:])

    pts_l_full, pts_r_full = analyzer._preprocess_and_get_matches(left_path, right_path)

    if pts_l_full is None or len(pts_l_full) == 0:
        print("未能在此图像上找到任何匹配点。请尝试其他图像索引。")
        return

    print(f"找到了 {len(pts_l_full)} 个匹配点。")

    # 3. 进行三角化和重投影
    print("正在使用你当前的 P1/P2 矩阵进行三维重建和重投影...")
    points_4d_hom = cv2.triangulatePoints(analyzer.calib['P1'], analyzer.calib['P2'], pts_l_full.T, pts_r_full.T)

    # 检查 w 坐标
    w_coords = points_4d_hom[3]
    if np.any(w_coords == 0):
        print("警告: 三角化过程中出现 w=0 的点，结果可能无效。")
        w_coords[w_coords == 0] = 1e-6

    reprojected_l_hom = (analyzer.calib['P1'] @ points_4d_hom)
    reprojected_l_w = reprojected_l_hom[2]
    if np.any(reprojected_l_w == 0):
        print("警告: 重投影过程中出现 w=0 的点，结果可能无效。")
        reprojected_l_w[reprojected_l_w == 0] = 1e-6

    reprojected_l_pts = (reprojected_l_hom[:2] / reprojected_l_w).T

    print("计算完成。")

    # 4. 可视化结果
    img_l_raw = cv2.imread(left_path)  # 加载彩色图以便于观察
    img_l_rect = cv2.remap(img_l_raw, analyzer.calib['map1_left'], analyzer.calib['map2_left'], cv2.INTER_LINEAR)

    for i in range(len(pts_l_full)):
        # 原始检测点 (绿色圆圈)
        pt_orig = (int(pts_l_full[i, 0]), int(pts_l_full[i, 1]))
        cv2.circle(img_l_rect, pt_orig, 6, (0, 255, 0), 2)  # Green circle

        # 重投影点 (红色十字)
        pt_reproj = (int(reprojected_l_pts[i, 0]), int(reprojected_l_pts[i, 1]))
        cv2.drawMarker(img_l_rect, pt_reproj, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)  # Red cross

    # 5. 显示图像和结论
    output_filename = "reprojection_debug_output.png"
    cv2.imwrite(output_filename, img_l_rect)
    print("\n" + "=" * 60)
    print("                   诊断结果")
    print("=" * 60)
    print(f"诊断图像已保存为: {output_filename}")
    print("请打开此图像并观察：")
    print("  - 绿色圆圈: 模型检测到的原始特征点位置。")
    print("  - 红色十字: 使用你的 P1/P2 重建三维点后再投影回来的位置。")
    print("\n--- 结论 ---")
    print("❌ 如果红色十字和绿色圆圈相距很远，则证明你的 P1/P2 是错误的。")
    print("✅ 如果红色十字精确地落在绿色圆圈的中心，则 P1/P2 是正确的。")
    print("\n--- 下一步 ---")
    print("如果诊断为错误，请立即运行 'create_rectified_calib.py' 来生成正确的标定文件。")
    print("=" * 60)

    cv2.imshow("Reprojection Visual Debugger", img_l_rect)
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_reprojection_error()
