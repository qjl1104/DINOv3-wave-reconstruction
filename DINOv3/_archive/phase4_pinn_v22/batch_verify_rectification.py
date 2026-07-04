import cv2
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------------------
# 配置区域
# ----------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 师兄的参数文件
CALIB_PATH = os.path.join(BASE_DIR, "calibration_paper_params.npz")

# 图片文件夹路径
LEFT_DIR = r"D:\Research\wave_reconstruction_project\data\left_images"
RIGHT_DIR = r"D:\Research\wave_reconstruction_project\data\right_images"

# 输出目录
OUT_DIR = os.path.join(BASE_DIR, "batch_test_output")
os.makedirs(OUT_DIR, exist_ok=True)


class BatchVerifier:
    def __init__(self, calib_path):
        self.calib_path = calib_path
        self.load_params()
        self.map1_l, self.map2_l = None, None
        self.map1_r, self.map2_r = None, None
        self.img_size = None  # 将在读取第一张图时初始化

    def load_params(self):
        if not os.path.exists(self.calib_path):
            print(f"[Error] 找不到参数文件: {self.calib_path}")
            return False

        data = np.load(self.calib_path)
        print(f"[Init] 加载参数: {self.calib_path}")

        # 提取 K, R, T (兼容不同命名)
        if 'K_left' in data:
            self.K1 = data['K_left'].astype(np.float64)
            self.D1 = data['D_left'].astype(np.float64)
            self.K2 = data['K_right'].astype(np.float64)
            self.D2 = data['D_right'].astype(np.float64)
        else:
            self.K1 = data['K1'].astype(np.float64)
            self.D1 = data['D1'].astype(np.float64)
            self.K2 = data['K2'].astype(np.float64)
            self.D2 = data['D2'].astype(np.float64)

        self.R = data['R'].astype(np.float64)
        self.T = data['T'].astype(np.float64).flatten()
        return True

    def init_rectification_maps(self, w, h):
        """根据图片尺寸和加载的 K,R,T 重新计算 Map"""
        if self.img_size == (w, h) and self.map1_l is not None:
            return  # 已经初始化过，且尺寸没变

        print(f"[Info] 正在为分辨率 {w}x{h} 生成矫正映射表...")
        self.img_size = (w, h)

        # alpha=0 裁剪黑边; alpha=1 保留全图. 这里用 alpha=0 试试 (更接近师兄可能的操作)
        # 或者为了保险用 alpha=-1 (OpenCV自动)
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, (w, h), self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
        )

        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, (w, h), cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2, (w, h), cv2.CV_32FC1)

    def calc_error(self, imgL, imgR):
        # 1. Remap
        rect_L = cv2.remap(imgL, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_R = cv2.remap(imgR, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        # 2. SIFT Match
        # 降采样加速计算 (计算误差不需要全分辨率)
        sift = cv2.SIFT_create()

        # 转换为灰度
        grayL = cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(grayL, None)
        kp2, des2 = sift.detectAndCompute(grayR, None)

        if des1 is None or des2 is None: return 999.0, rect_L, rect_R

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)

        ptsL, ptsR = [], []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                ptsL.append(kp1[m.queryIdx].pt)
                ptsR.append(kp2[m.trainIdx].pt)

        if len(ptsL) < 10: return 999.0, rect_L, rect_R

        # 3. Y-diff Mean
        ptsL = np.array(ptsL)
        ptsR = np.array(ptsR)
        y_diff = np.abs(ptsL[:, 1] - ptsR[:, 1])
        return np.mean(y_diff), rect_L, rect_R

    def run_batch_test(self):
        # 1. 扫描文件
        exts = ['*.bmp', '*.jpg', '*.png', '*.tiff']
        left_files = []
        for ext in exts:
            left_files.extend(glob.glob(os.path.join(LEFT_DIR, ext)))

        print(f"[Scan] 找到 {len(left_files)} 张左视图。开始匹配右视图...")

        results = []

        for l_path in tqdm(left_files):
            filename = os.path.basename(l_path)
            # 假设命名规则: leftXXXX.bmp -> rightXXXX.bmp
            # 或者 left_01.jpg -> right_01.jpg
            r_filename = filename.replace("left", "right").replace("Left", "Right")
            r_path = os.path.join(RIGHT_DIR, r_filename)

            if not os.path.exists(r_path):
                # 尝试另一种命名规则 (如果有)
                continue

            # 读取图片
            imgL = cv2.imread(l_path)
            imgR = cv2.imread(r_path)
            if imgL is None or imgR is None: continue

            h, w = imgL.shape[:2]

            # 初始化 Maps (只在第一次或尺寸变动时运行)
            self.init_rectification_maps(w, h)

            # 计算误差
            err, rL, rR = self.calc_error(imgL, imgR)

            results.append({
                "Filename": filename,
                "Error": err,
                "Matches": "OK" if err < 100 else "Few/None"
            })

            # 如果发现极好的结果，保存下来看看
            if err < 1.5:
                cv2.imwrite(os.path.join(OUT_DIR, f"GOOD_{filename}"), np.hstack([rL, rR]))

            # 只测试前 20 张 (如果图片太多，避免跑太久)
            # if len(results) >= 20: break

        # 结果汇总
        df = pd.DataFrame(results)
        if df.empty:
            print("没有找到匹配的图片对。")
            return

        df = df.sort_values(by="Error")
        print("\n" + "=" * 50)
        print("   🏆 最佳匹配结果 (Top 5)")
        print("=" * 50)
        print(df.head(5))

        best_err = df.iloc[0]["Error"]
        if best_err < 1.5:
            print(f"\n✅ 发现完美匹配！图片 [{df.iloc[0]['Filename']}] 的误差仅为 {best_err:.2f} px。")
            print("   结论：标定参数是有效的，但仅针对特定的图片（可能是相机位移前的图片）。")
        else:
            print(f"\n❌ 所有图片的误差都很大 (最小误差: {best_err:.2f} px)。")
            print("   结论：这套标定参数【完全不适用于】当前的整个数据集。")
            print("   建议使用 auto_rectify_v3.py 生成的新参数。")


if __name__ == "__main__":
    if os.path.exists(CALIB_PATH):
        verifier = BatchVerifier(CALIB_PATH)
        verifier.run_batch_test()
    else:
        print("请检查标定文件路径。")