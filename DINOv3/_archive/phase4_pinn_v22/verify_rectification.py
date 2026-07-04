import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------------------
# 配置区域
# ----------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 你刚刚上传的图片文件名
TARGET_IMG_NAME = "image_8288fd.jpg"
TARGET_IMG_PATH = os.path.join(BASE_DIR, TARGET_IMG_NAME)


def analyze_rectification_quality():
    if not os.path.exists(TARGET_IMG_PATH):
        print(f"❌ 找不到图片: {TARGET_IMG_PATH}")
        print("请确保你已经把上传的图片保存在当前脚本同级目录下。")
        return

    print(f"[读取] 正在分析图片: {TARGET_IMG_NAME} ...")
    img = cv2.imread(TARGET_IMG_PATH)
    if img is None:
        print("❌ 图片读取失败。")
        return

    h, w = img.shape[:2]
    print(f"   尺寸: {w}x{h}")

    # 1. 假设这是矫正后的拼接图，将其切分为左右两半
    # 注意：如果图片包含边框或文字，可能需要裁剪。这里假设是纯拼接图。
    mid_w = w // 2
    imgL = img[:, :mid_w]
    imgR = img[:, mid_w:]

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # 2. 基于区域的密集匹配 (Template Matching)
    # 这种方法不需要特征点，对波浪这种连续纹理非常有效
    print("[分析] 正在进行密集区域匹配 (Template Matching)...")

    patch_size = 31  # 匹配块大小
    search_range_y = 20  # 在上下 20 像素范围内搜索
    search_range_x = 100  # 在左右 100 像素范围内搜索 (视差范围)

    # 网格采样点
    grid_y = np.linspace(patch_size, h - patch_size, 20, dtype=int)
    grid_x = np.linspace(patch_size, mid_w - patch_size - search_range_x, 20, dtype=int)

    y_errors = []
    valid_points = 0

    vis_img = img.copy()

    for y in grid_y:
        for x in grid_x:
            # 提取左图 Patch
            patch = grayL[y - patch_size // 2: y + patch_size // 2 + 1,
                    x - patch_size // 2: x + patch_size // 2 + 1]

            # 计算局部方差，忽略平坦区域（无纹理区域匹配不准）
            if np.std(patch) < 5.0:
                continue

            # 在右图对应区域搜索
            # 限制 Y 搜索范围 (理应在同一行，即 y_start = y)
            y_min = max(0, y - search_range_y)
            y_max = min(h, y + search_range_y)

            # 限制 X 搜索范围 (右图对应点通常在左侧，视差 d > 0)
            # x_right = x - d. 我们在 x 附近搜索
            x_min = max(0, x - search_range_x)
            x_max = min(mid_w, x + 20)  # 允许少量右偏

            search_region = grayR[y_min:y_max, x_min:x_max]

            if search_region.shape[0] < patch.shape[0] or search_region.shape[1] < patch.shape[1]:
                continue

            # 模板匹配
            res = cv2.matchTemplate(search_region, patch, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # 只有相关性足够高才算匹配成功
            if max_val > 0.8:
                # 还原到全局坐标
                match_x = x_min + max_loc[0] + patch_size // 2
                match_y = y_min + max_loc[1] + patch_size // 2

                # 计算垂直误差
                y_diff = abs(match_y - y)
                y_errors.append(y_diff)
                valid_points += 1

                # 在图上画线可视化 (红色: 左点, 绿色: 右点连线)
                # 左图点 (x, y)
                # 右图点 (mid_w + match_x, match_y)
                color = (0, 255, 0) if y_diff < 2 else (0, 0, 255)
                cv2.line(vis_img, (x, y), (mid_w + match_x, match_y), color, 1)
                cv2.circle(vis_img, (x, y), 2, (0, 255, 255), -1)

    print(f"   -> 成功匹配 {valid_points} 个区域块")

    if valid_points == 0:
        print("❌ 无法找到可靠的匹配区域，图片可能完全没有纹理或错位太严重。")
    else:
        mean_error = np.mean(y_errors)
        max_error = np.max(y_errors)

        print(f"\n📊 [深度诊断结果]")
        print(f"   平均垂直误差: {mean_error:.2f} px")
        print(f"   最大垂直误差: {max_error:.2f} px")

        if mean_error < 1.5:
            print("✅ 结论: 矫正非常成功！(误差 < 1.5 px)")
            print("   波浪纹理已水平对齐，可以直接用于波高重建。")
        elif mean_error < 3.0:
            print("⚠️ 结论: 矫正尚可 (误差 < 3.0 px)。")
            print("   对于大波浪可以接受，微小波纹可能会有噪点。")
        else:
            print("❌ 结论: 矫正失败 (误差较大)。")
            print("   请检查是否左右图片反了，或者标定板拍摄问题。")

        # 保存分析图
        out_path = os.path.join(BASE_DIR, "analysis_result.jpg")
        cv2.imwrite(out_path, vis_img)
        print(f"🖼️ 分析详情图已保存: {out_path}")

        # 显示
        scale = 1000 / vis_img.shape[1]
        vis_small = cv2.resize(vis_img, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Rectification Analysis", vis_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_rectification_quality()