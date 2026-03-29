import cv2
import numpy as np
import os
import glob


def generate_clean_params():
    # 1. 锁定源文件：使用最原始的 Matlab 导出文件
    # 这个文件的 R 和 T 是经过验证最靠谱的 (虽然它的 Map 是 int16，但我们只取 R, T, K, D)
    source_file = r'D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_from_matlab_full.npz'
    output_file = 'paper_params_recalculated.npz'

    print(f"--- 步骤 1: 读取原始数据源 {source_file} ---")
    if not os.path.exists(source_file):
        print(f"错误: 找不到文件 {source_file}")
        return

    data = np.load(source_file)

    # 提取基础内参 (Intrinsic) 和 畸变系数 (Distortion)
    # 注意：旧文件键名为 K_left, D_left 等
    K1 = data['K_left']
    D1 = data['D_left']
    K2 = data['K_right']
    D2 = data['D_right']

    # 提取基础外参 (Extrinsic)
    R = data['R']
    T = data['T']

    # 图像物理分辨率 (必须确认是 2560x1600)
    image_size = tuple(data['image_size'])  # 通常是 [2560, 1600]

    print(f"原始相机分辨率: {image_size}")
    print(f"原始基线 T: {T}")
    print(f"原始旋转 R (前3个值): {R[0]}")
    # 验证 R 没有被转置: R[0,1] 应该是负数 (-0.141)
    if R[0, 1] > 0:
        print("⚠️ 警告: 检测到源文件 R 矩阵可能被转置！正在修正...")
        R = R.T
    else:
        print("√ R 矩阵方向正确 (未转置)")

    print("\n--- 步骤 2: 执行高精度立体矫正 (Stereo Rectify) ---")
    # 关键设置：
    # alpha = 1: 保留所有像素 (All Pixels)。
    #   -> 优点: 视野最大，左右图重叠区最大，容易匹配。
    #   -> 缺点: 边缘会有黑色无效区域 (这是正常的)。
    #   -> 结果: 焦距 f 会变小 (约 2600~3000)，视差 d 变小，容易计算深度。

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1,
        newImageSize=image_size
    )

    # 检查生成的 Q 矩阵焦距
    f_new = Q[2, 3]
    print(f"生成的新 Q 矩阵焦距 f = {f_new:.2f}")

    if f_new > 5000:
        print("❌ 错误: 焦距依然过大，可能是 alpha 设置未生效或计算错误。")
    elif f_new < 2000:
        print("⚠️ 警告: 焦距过小，图像可能被过度缩小。")
    else:
        print("√ 焦距在合理范围 (2000-4000)，视野正常，重叠区将恢复。")

    print("\n--- 步骤 3: 生成高精度映射表 (Float32 Maps) ---")
    # 使用 CV_32FC1 生成浮点数映射表，确保亚像素精度，避免波浪纹理出现断层
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    print("\n--- 步骤 4: 保存纯净的新文件 ---")
    # 保存所有必要的参数，供后续 Inference 和 Training 脚本直接使用
    np.savez(output_file,
             # 基础参数
             K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
             # 矫正参数
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             # 映射表
             map1_left=map1_left, map2_left=map2_left,
             map1_right=map1_right, map2_right=map2_right,
             # 有效区域 (用于去黑边，如果需要的话)
             roi_left=roi1, roi_right=roi2)

    print(f"成功保存至: {output_file}")
    print("请更新您的 Inference 和 Training 脚本以使用此文件。")


def verify_batch(npz_file, left_dir, right_dir, num_samples=5):
    print(f"\n--- 步骤 5: 批量验证矫正效果 (采样 {num_samples} 组) ---")
    if not os.path.exists(npz_file):
        print(f"错误: 找不到标定文件 {npz_file}")
        return

    data = np.load(npz_file)
    map1_l, map2_l = data['map1_left'], data['map2_left']
    map1_r, map2_r = data['map1_right'], data['map2_right']

    # 搜索图片
    # 支持 bmp, jpg, png
    exts = ['*.bmp', '*.jpg', '*.png']
    left_files = []
    for ext in exts:
        left_files.extend(glob.glob(os.path.join(left_dir, ext)))

    left_files = sorted(left_files)
    if not left_files:
        print(f"在 {left_dir} 未找到图片，无法验证。")
        return

    print(f"找到 {len(left_files)} 张左图，正在处理前 {num_samples} 张...")

    count = 0
    for l_path in left_files:
        if count >= num_samples: break

        basename = os.path.basename(l_path)
        # 简单推断右图文件名: left -> right (兼容大小写)
        if "left" in basename:
            r_basename = basename.replace("left", "right")
        elif "Left" in basename:
            r_basename = basename.replace("Left", "Right")
        else:
            # 如果文件名没有left，尝试找同名文件
            r_basename = basename

        r_path = os.path.join(right_dir, r_basename)

        if not os.path.exists(r_path):
            print(f"跳过: 找不到对应的右图 {r_basename}")
            continue

        imgL = cv2.imread(l_path)
        imgR = cv2.imread(r_path)

        if imgL is None or imgR is None: continue

        # 矫正
        rectL = cv2.remap(imgL, map1_l, map2_l, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map1_r, map2_r, cv2.INTER_LINEAR)

        # 1. 并排显示 (Side-by-Side) - 检查行对齐
        # 画绿色水平线
        vis_side = np.hstack((rectL, rectR))
        for y in range(0, vis_side.shape[0], 100):
            cv2.line(vis_side, (0, y), (vis_side.shape[1], y), (0, 255, 0), 2)

        # 2. 叠加显示 (Blend) - 检查左右重叠
        vis_blend = cv2.addWeighted(rectL, 0.5, rectR, 0.5, 0)
        for y in range(0, vis_blend.shape[0], 100):
            cv2.line(vis_blend, (0, y), (vis_blend.shape[1], y), (0, 255, 0), 1)

        # 保存结果
        s_name = f"verify_S{count + 1}_{basename}_side.jpg"
        b_name = f"verify_S{count + 1}_{basename}_blend.jpg"
        cv2.imwrite(s_name, vis_side)
        cv2.imwrite(b_name, vis_blend)
        print(f"  [组{count + 1}] 生成完毕: {s_name} (对齐) | {b_name} (重叠)")
        count += 1

    print("\n验证完成！请查看生成的 .jpg 图片。")
    print("判定标准：")
    print("1. Side图中的绿线必须水平穿过左右图相同的特征点 (行对齐)。")
    print("2. Blend图中应该能看到成对的波浪重影 (说明有重叠视野)。")


if __name__ == "__main__":
    generate_clean_params()

    # 这里填入您实际的图片文件夹路径
    l_dir = r"D:\Research\wave_reconstruction_project\data\left_images"
    r_dir = r"D:\Research\wave_reconstruction_project\data\right_images"

    # 自动执行批量验证
    verify_batch('paper_params_recalculated.npz', l_dir, r_dir, num_samples=5)