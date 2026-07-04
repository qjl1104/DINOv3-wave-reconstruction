# particle_processing/00_generate_background.py
import cv2
import numpy as np
import glob
import os
import random


def generate_background_model(image_dir, output_path, num_images_to_sample=50):
    """
    通过计算多张图像的中位数来生成一个静态的背景模型。

    Args:
        image_dir (str): 包含原始图像的文件夹路径。
        output_path (str): 生成的背景图像的保存路径。
        num_images_to_sample (int): 用于计算中位数的随机抽样图像数量。
    """
    print(f"正在从 '{image_dir}' 生成背景模型...")

    image_files = sorted(glob.glob(os.path.join(image_dir, '*.bmp')))
    if not image_files:
        print(f"错误: 在 '{image_dir}' 中未找到任何图像。")
        return

    # 从所有图像中随机抽取一部分样本，以加快处理速度
    if len(image_files) > num_images_to_sample:
        sampled_files = random.sample(image_files, num_images_to_sample)
    else:
        sampled_files = image_files
    print(f"将使用 {len(sampled_files)} 张样本图像。")

    # 读取所有样本图像并存入一个列表中
    images = []
    for file in sampled_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    if not images:
        print("错误: 无法读取任何样本图像。")
        return

    # 将图像列表堆叠成一个三维数组 (height, width, num_images)
    image_stack = np.stack(images, axis=-1)

    # 沿着图像序列的轴计算中位数，这会有效地移除移动的物体（粒子）
    median_frame = np.median(image_stack, axis=-1).astype(np.uint8)

    # 保存生成的背景模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, median_frame)
    print(f"背景模型已成功保存至: {output_path}")


if __name__ == '__main__':
    # --- 用户需要配置的路径 ---
    # 包含原始粒子图像的文件夹
    raw_left_dir = "../data/left_images/"
    raw_right_dir = "../data/right_images/"

    # 生成的背景图像的保存路径
    # 我们将它保存在 preprocessed 文件夹的根目录
    out_bg_left = "../data/preprocessed/background_left.png"
    out_bg_right = "../data/preprocessed/background_right.png"

    # --- 执行主函数 ---
    generate_background_model(raw_left_dir, out_bg_left)
    generate_background_model(raw_right_dir, out_bg_right)
