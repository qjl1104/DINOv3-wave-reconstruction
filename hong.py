"""
圆片检测调试工具
用于找到最佳的圆片检测参数
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def test_circle_detection():
    """
    交互式测试圆片检测参数
    """
    # 设置图像路径
    left_dir = r"D:\zuchuan\lresult"
    right_dir = r"D:\zuchuan\rresult"

    # 获取第一张图像
    left_path = Path(left_dir)
    image_files = list(left_path.glob('*.bmp'))
    if not image_files:
        image_files = list(left_path.glob('*.jpg'))

    if not image_files:
        print("未找到图像文件")
        return

    # 读取第一张图像
    image_path = str(image_files[0])
    print(f"使用图像: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return

    print(f"图像尺寸: {image.shape}")

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 测试不同的预处理方法
    print("\n测试不同的预处理方法...")

    # 方法1: 直接使用霍夫圆检测
    print("\n方法1: 霍夫圆检测")
    test_hough_circles(image, gray)

    # 方法2: 使用Blob检测
    print("\n方法2: Blob检测")
    test_blob_detection(image, gray)

    # 方法3: 使用轮廓检测
    print("\n方法3: 轮廓检测")
    test_contour_detection(image, gray)

    # 方法4: 使用自适应阈值
    print("\n方法4: 自适应阈值 + 霍夫圆")
    test_adaptive_threshold(image, gray)

def test_hough_circles(image, gray):
    """测试霍夫圆检测"""
    # 高斯滤波
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 测试不同的参数组合
    param_sets = [
        {'min_r': 5, 'max_r': 20, 'param1': 50, 'param2': 20},
        {'min_r': 10, 'max_r': 30, 'param1': 100, 'param2': 30},
        {'min_r': 15, 'max_r': 40, 'param1': 150, 'param2': 40},
        {'min_r': 20, 'max_r': 50, 'param1': 200, 'param2': 50},
        {'min_r': 8, 'max_r': 25, 'param1': 80, 'param2': 25},
    ]

    best_count = 0
    best_params = None

    for params in param_sets:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=params['param1'],
            param2=params['param2'],
            minRadius=params['min_r'],
            maxRadius=params['max_r']
        )

        count = 0 if circles is None else len(circles[0])
        print(f"  参数 {params}: 检测到 {count} 个圆")

        if count > best_count:
            best_count = count
            best_params = params
            best_circles = circles

    print(f"  最佳参数: {best_params}, 检测到 {best_count} 个圆")

    # 可视化最佳结果
    if best_count > 0:
        result = image.copy()
        circles = np.round(best_circles[0, :]).astype("int")
        for (x, y, r) in circles[:100]:  # 只画前100个
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

        # 保存结果
        cv2.imwrite('hough_circles_result.jpg', result)
        print("  结果已保存到 hough_circles_result.jpg")

        # 显示缩小版本
        show_image = cv2.resize(result, (1280, 800))
        cv2.imshow('Hough Circles', show_image)
        cv2.waitKey(1000)

    return best_params, best_count

def test_blob_detection(image, gray):
    """测试Blob检测"""
    # 设置Blob检测参数
    params = cv2.SimpleBlobDetector_Params()

    # 阈值
    params.minThreshold = 10
    params.maxThreshold = 200

    # 面积
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 3000

    # 圆度
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # 凸度
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # 惯性
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    # 创建检测器
    detector = cv2.SimpleBlobDetector_create(params)

    # 检测
    keypoints = detector.detect(gray)
    print(f"  Blob检测: 发现 {len(keypoints)} 个blob")

    if len(keypoints) > 0:
        # 分析大小
        sizes = [kp.size for kp in keypoints]
        print(f"  大小范围: [{min(sizes):.1f}, {max(sizes):.1f}]")
        print(f"  平均大小: {np.mean(sizes):.1f}")

        # 可视化
        result = cv2.drawKeypoints(
            image, keypoints, None,
            (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        cv2.imwrite('blob_detection_result.jpg', result)
        print("  结果已保存到 blob_detection_result.jpg")

        # 显示缩小版本
        show_image = cv2.resize(result, (1280, 800))
        cv2.imshow('Blob Detection', show_image)
        cv2.waitKey(1000)

        # 基于blob大小推荐霍夫圆参数
        recommended_min_r = int(min(sizes) * 0.4)
        recommended_max_r = int(max(sizes) * 0.6)
        print(f"\n  基于Blob检测推荐的霍夫圆参数:")
        print(f"    min_radius: {recommended_min_r}")
        print(f"    max_radius: {recommended_max_r}")

        return recommended_min_r, recommended_max_r

    return None, None

def test_contour_detection(image, gray):
    """测试轮廓检测"""
    # 应用阈值
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 筛选圆形轮廓
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50 or area > 3000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.5:  # 圆度阈值
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append((int(x), int(y), int(radius)))

    print(f"  轮廓检测: 发现 {len(circles)} 个圆形轮廓")

    if len(circles) > 0:
        # 可视化
        result = image.copy()
        for (x, y, r) in circles[:100]:
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

        cv2.imwrite('contour_detection_result.jpg', result)
        print("  结果已保存到 contour_detection_result.jpg")

        # 显示缩小版本
        show_image = cv2.resize(result, (1280, 800))
        cv2.imshow('Contour Detection', show_image)
        cv2.waitKey(1000)

    return circles

def test_adaptive_threshold(image, gray):
    """测试自适应阈值"""
    # 应用自适应阈值
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

    # 在处理后的图像上检测圆
    circles = cv2.HoughCircles(
        opened,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=20,
        minRadius=10,
        maxRadius=30
    )

    count = 0 if circles is None else len(circles[0])
    print(f"  自适应阈值 + 霍夫圆: 检测到 {count} 个圆")

    if count > 0:
        result = image.copy()
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles[:100]:
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

        cv2.imwrite('adaptive_threshold_result.jpg', result)
        print("  结果已保存到 adaptive_threshold_result.jpg")

        # 显示缩小版本
        show_image = cv2.resize(result, (1280, 800))
        cv2.imshow('Adaptive Threshold', show_image)
        cv2.waitKey(1000)

    return count

def analyze_first_n_frames(n=10):
    """
    分析前n帧图像，统计圆片检测结果
    """
    left_dir = r"D:\zuchuan\lresult"
    left_path = Path(left_dir)

    image_files = list(left_path.glob('*.bmp'))[:n]
    if not image_files:
        image_files = list(left_path.glob('*.jpg'))[:n]

    print(f"分析前 {len(image_files)} 帧图像...")

    # 测试最佳参数
    best_params = {'min_r': 10, 'max_r': 30, 'param1': 100, 'param2': 30}

    results = []
    for i, image_file in enumerate(image_files):
        image = cv2.imread(str(image_file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=best_params['param1'],
            param2=best_params['param2'],
            minRadius=best_params['min_r'],
            maxRadius=best_params['max_r']
        )

        count = 0 if circles is None else len(circles[0])
        results.append(count)
        print(f"  帧 {i}: 检测到 {count} 个圆")

    print(f"\n统计结果:")
    print(f"  平均检测数: {np.mean(results):.1f}")
    print(f"  最小检测数: {min(results)}")
    print(f"  最大检测数: {max(results)}")
    print(f"  标准差: {np.std(results):.1f}")

def generate_detection_report():
    """
    生成完整的检测报告
    """
    print("=" * 50)
    print("圆片检测参数优化报告")
    print("=" * 50)

    # 测试单帧
    print("\n1. 单帧检测测试")
    test_circle_detection()

    # 分析多帧
    print("\n2. 多帧统计分析")
    analyze_first_n_frames(10)

    print("\n3. 推荐参数")
    print("根据分析结果，推荐使用以下参数:")
    print("CircleDetector(")
    print("    min_radius=10,")
    print("    max_radius=30,")
    print("    min_dist=30")
    print(")")

    print("\n报告生成完成！")
    print("请查看生成的图像文件:")
    print("  - hough_circles_result.jpg")
    print("  - blob_detection_result.jpg")
    print("  - contour_detection_result.jpg")
    print("  - adaptive_threshold_result.jpg")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_detection_report()