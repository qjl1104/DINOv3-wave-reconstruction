"""
测试轮廓检测方法的效果
快速验证新的检测算法
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class ImprovedCircleDetector:
    """改进的圆片检测器"""

    def __init__(self, min_radius=5, max_radius=50, min_area=50, max_area=3000,
                 circularity_threshold=0.4):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold

    def detect_contour(self, image):
        """轮廓检测方法"""
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 使用自适应阈值
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # 面积筛选
            if area < self.min_area or area > self.max_area:
                continue

            # 计算圆度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 圆度筛选
            if circularity > self.circularity_threshold:
                (x, y), radius = cv2.minEnclosingCircle(contour)

                # 半径筛选
                if self.min_radius <= radius <= self.max_radius:
                    circles.append([int(x), int(y), int(radius)])

        return np.array(circles), binary

    def detect_combined(self, image):
        """组合检测方法"""
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 方法1: 轮廓检测
        circles_contour, binary = self.detect_contour(image)

        # 方法2: 霍夫圆检测在二值图上
        circles_hough = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=15,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles_hough is not None:
            circles_hough = np.round(circles_hough[0, :]).astype("int")
        else:
            circles_hough = np.array([])

        # 合并结果（去重）
        all_circles = []
        if len(circles_contour) > 0:
            all_circles.extend(circles_contour.tolist())

        if len(circles_hough) > 0:
            for ch in circles_hough:
                # 检查是否重复
                is_duplicate = False
                for ac in all_circles:
                    dist = np.sqrt((ch[0] - ac[0]) ** 2 + (ch[1] - ac[1]) ** 2)
                    if dist < 10:  # 距离阈值
                        is_duplicate = True
                        break
                if not is_duplicate:
                    all_circles.append(ch.tolist())

        return np.array(all_circles) if all_circles else np.array([])


def test_on_single_image():
    """测试单张图像"""
    # 图像路径
    left_dir = r"D:\zuchuan\lresult"
    image_path = Path(left_dir) / "lresult0001.bmp"

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    print(f"图像尺寸: {image.shape}")

    # 创建检测器
    detector = ImprovedCircleDetector(
        min_radius=5,
        max_radius=50,
        min_area=50,
        max_area=3000,
        circularity_threshold=0.3  # 进一步降低阈值
    )

    # 测试轮廓检测
    print("\n测试轮廓检测方法...")
    circles_contour, binary = detector.detect_contour(image)
    print(f"轮廓检测: 发现 {len(circles_contour)} 个圆片")

    # 测试组合方法
    print("\n测试组合检测方法...")
    circles_combined = detector.detect_combined(image)
    print(f"组合检测: 发现 {len(circles_combined)} 个圆片")

    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原图
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    # 二值图
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('二值化图像')
    axes[0, 1].axis('off')

    # 轮廓检测结果
    result_contour = image.copy()
    for (x, y, r) in circles_contour[:200]:  # 最多画200个
        cv2.circle(result_contour, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result_contour, (x, y), 2, (0, 0, 255), 3)
    axes[0, 2].imshow(cv2.cvtColor(result_contour, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'轮廓检测 ({len(circles_contour)} 个)')
    axes[0, 2].axis('off')

    # 组合检测结果
    result_combined = image.copy()
    for (x, y, r) in circles_combined[:200]:
        cv2.circle(result_combined, (x, y), r, (255, 0, 0), 2)
        cv2.circle(result_combined, (x, y), 2, (0, 255, 0), 3)
    axes[1, 0].imshow(cv2.cvtColor(result_combined, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'组合检测 ({len(circles_combined)} 个)')
    axes[1, 0].axis('off')

    # 局部放大
    if len(circles_contour) > 0:
        # 选择一个区域进行放大
        cx, cy = int(image.shape[1] / 2), int(image.shape[0] / 2)
        roi_size = 300
        x1, y1 = max(0, cx - roi_size), max(0, cy - roi_size)
        x2, y2 = min(image.shape[1], cx + roi_size), min(image.shape[0], cy + roi_size)

        roi = result_contour[y1:y2, x1:x2]
        axes[1, 1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('局部放大')
        axes[1, 1].axis('off')

    # 统计信息
    axes[1, 2].text(0.1, 0.8, f'轮廓检测: {len(circles_contour)} 个圆片', fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'组合检测: {len(circles_combined)} 个圆片', fontsize=12)

    if len(circles_contour) > 0:
        radii = circles_contour[:, 2]
        axes[1, 2].text(0.1, 0.4, f'半径范围: {radii.min()}-{radii.max()} pixels', fontsize=12)
        axes[1, 2].text(0.1, 0.2, f'平均半径: {radii.mean():.1f} pixels', fontsize=12)

    axes[1, 2].set_title('统计信息')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('improved_detection_results.png', dpi=150)
    plt.show()

    print("\n结果已保存到 improved_detection_results.png")

    # 保存检测结果图像
    cv2.imwrite('contour_detection_result_improved.jpg', result_contour)
    cv2.imwrite('combined_detection_result.jpg', result_combined)

    return circles_combined


def test_stereo_matching():
    """测试双目匹配"""
    left_dir = r"D:\zuchuan\lresult"
    right_dir = r"D:\zuchuan\rresult"

    # 读取第一对图像
    left_image = cv2.imread(str(Path(left_dir) / "lresult0001.bmp"))
    right_image = cv2.imread(str(Path(right_dir) / "rresult0001.bmp"))

    if left_image is None or right_image is None:
        print("无法读取双目图像")
        return

    # 创建检测器
    detector = ImprovedCircleDetector(
        min_radius=5,
        max_radius=50,
        min_area=50,
        max_area=3000,
        circularity_threshold=0.3
    )

    # 检测圆片
    print("检测左图圆片...")
    left_circles, _ = detector.detect_contour(left_image)
    print(f"左图: {len(left_circles)} 个圆片")

    print("检测右图圆片...")
    right_circles, _ = detector.detect_contour(right_image)
    print(f"右图: {len(right_circles)} 个圆片")

    # 简单匹配（基于y坐标相似性）
    matches = []
    for i, lc in enumerate(left_circles):
        best_match = None
        min_y_diff = float('inf')

        for j, rc in enumerate(right_circles):
            y_diff = abs(lc[1] - rc[1])  # y坐标差异

            if y_diff < 10 and y_diff < min_y_diff:  # 极线约束
                # 检查视差是否合理
                disparity = lc[0] - rc[0]
                if 10 < disparity < 200:  # 视差范围
                    min_y_diff = y_diff
                    best_match = j

        if best_match is not None:
            matches.append((i, best_match))

    print(f"\n匹配结果: {len(matches)} 对匹配")

    # 可视化匹配
    combined = np.hstack([left_image, right_image])
    result = combined.copy()

    # 画出匹配线
    for i, (li, ri) in enumerate(matches[:50]):  # 最多画50条
        lc = left_circles[li]
        rc = right_circles[ri]

        # 左图圆
        cv2.circle(result, (lc[0], lc[1]), lc[2], (0, 255, 0), 2)
        # 右图圆（x坐标需要偏移）
        cv2.circle(result, (rc[0] + left_image.shape[1], rc[1]), rc[2], (0, 255, 0), 2)

        # 连线
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.line(result, (lc[0], lc[1]),
                 (rc[0] + left_image.shape[1], rc[1]), color, 1)

    # 保存结果
    cv2.imwrite('stereo_matching_result.jpg', result)
    print("匹配结果已保存到 stereo_matching_result.jpg")

    # 显示缩小版
    show_result = cv2.resize(result, (1920, 600))
    cv2.imshow('Stereo Matching', show_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("=" * 50)
    print("改进的圆片检测测试")
    print("=" * 50)

    # 测试单图检测
    print("\n1. 测试单图圆片检测")
    circles = test_on_single_image()

    if circles is not None and len(circles) > 100:
        print(f"\n检测效果良好！发现 {len(circles)} 个圆片")
        print("建议使用轮廓检测方法进行训练")
    else:
        print("\n检测效果需要改进")
        print("建议调整参数或尝试其他预处理方法")

    # 测试双目匹配
    print("\n2. 测试双目匹配")
    test_stereo_matching()

    print("\n测试完成！")
    print("请查看生成的图像文件:")
    print("  - improved_detection_results.png (检测结果对比)")
    print("  - contour_detection_result_improved.jpg (轮廓检测)")
    print("  - combined_detection_result.jpg (组合检测)")
    print("  - stereo_matching_result.jpg (双目匹配)")


if __name__ == "__main__":
    main()