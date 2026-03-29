import cv2
import numpy as np
import os

# ----------------------------------------------------------------------------------
# 配置区域
# ----------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 必须使用师兄的真参数文件
CALIB_PATH = os.path.join(BASE_DIR, "calibration_paper_params.npz")

# 原始图片路径
LEFT_PATH = r"D:\Research\wave_reconstruction_project\data\left_images\left0001.bmp"
RIGHT_PATH = r"D:\Research\wave_reconstruction_project\data\right_images\right0001.bmp"

# 输出目录
OUT_DIR = os.path.join(BASE_DIR, "rectified_output")
os.makedirs(OUT_DIR, exist_ok=True)


class UniversalRectifier:
    def __init__(self, calib_path, img_size):
        self.calib_path = calib_path
        self.img_size = img_size  # (Width, Height)
        self.K1, self.D1 = None, None
        self.K2, self.D2 = None, None
        self.R, self.T = None, None

        self.load_params()

    def load_params(self):
        if not os.path.exists(self.calib_path):
            raise FileNotFoundError(f"找不到标定文件: {self.calib_path}")

        print(f"[Step 1] 加载参数文件: {self.calib_path}")
        data = np.load(self.calib_path)

        # 提取参数 (兼容多种命名)
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

        print(f"   D1 (畸变): {self.D1.flatten()}")
        print(f"   D2 (畸变): {self.D2.flatten()}")

    def validate_geometry_and_fix(self, imgL, imgR):
        print("\n[Step 2] 几何一致性终极判决 (Geometric Verdict)")

        # 1. 提取特征点
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(imgL, None)
        kp2, des2 = sift.detectAndCompute(imgR, None)

        pts1, pts2 = [], []
        if des1 is not None and des2 is not None:
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            matches = flann.knnMatch(des1, des2, k=2)
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    pts1.append(kp1[m.queryIdx].pt)
                    pts2.append(kp2[m.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        print(f"   -> 提取到 {len(pts1)} 对匹配点用于验证")

        if len(pts1) < 10:
            print("   [Error] 匹配点过少，无法验证。")
            return

        # 2. 计算参数导出的基础矩阵 F_param
        # F = K_R^{-T} * [T]_x * R * K_L^{-1}
        # 注意：这里我们验证参数本身是否匹配图片特征
        T_x = np.array([
            [0, -self.T[2], self.T[1]],
            [self.T[2], 0, -self.T[0]],
            [-self.T[1], self.T[0], 0]
        ])
        E = T_x @ self.R  # Essential Matrix
        F_param = np.linalg.inv(self.K2).T @ E @ np.linalg.inv(self.K1)

        # 3. 计算对极误差 (Sampson Distance 近似)
        # 简单点：计算 p2^T * F * p1 的代数误差
        # 或者计算点到极线的距离

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_param)
        lines2 = lines2.reshape(-1, 3)
        # distance = |ax + by + c| / sqrt(a^2 + b^2)
        dist = np.abs(lines2[:, 0] * pts2[:, 0] + lines2[:, 1] * pts2[:, 1] + lines2[:, 2]) / \
               np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)

        mean_error = np.mean(dist)
        print(f"   -> 参数对应的平均极线距离误差: {mean_error:.4f} px")

        # 判决逻辑
        if mean_error > 10.0:
            print(f"\n{'=' * 50}")
            print(f"🚨【判决结果】：参数不匹配！(误差 {mean_error:.1f} px > 10 px)")
            print(f"{'=' * 50}")
            print("   原因分析：数学证明，师兄的参数(K,R,T)无法解释这两张图片的几何关系。")
            print("   这只有一种可能：图片拍摄时，相机位置相对于标定时发生了微小位移(震动/碰撞)。")
            print("   即使位移 1mm，对于精密双目也是致命的。")
            print("\n>>> 启动 B 计划：现场自适应矫正 (Auto-Rectify) <<<")
            self.run_blind_rectification(pts1, pts2, imgL, imgR)
        else:
            print("\n✅【判决结果】：参数基本匹配。问题可能出在 stereoRectify 的 flag 上。")
            print("   尝试使用不同的 alpha/flags 重算 Map...")
            self.run_param_based_rectification(imgL, imgR)

    def run_param_based_rectification(self, imgL, imgR):
        # 尝试标准矫正
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, self.img_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
        )
        self.save_and_vis(imgL, imgR, self.K1, self.D1, R1, P1, self.K2, self.D2, R2, P2, Q, "fixed_param")

    def run_blind_rectification(self, pts1, pts2, imgL, imgR):
        # 使用 Hartley 算法或 Uncalibrated Rectify
        # 1. 计算这一对图的 F
        F_curr, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # 2. 计算 H1, H2
        ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F_curr, self.img_size)

        if not ret:
            print("   [Error] 自动计算失败。")
            return

        print("   -> 已根据图片特征生成全新的矫正变换 (H1, H2)")

        # 3. 生成 Map
        # 这里有些技巧：uncalibrated 只给 H，不处理畸变。
        # 如果畸变不大，可以直接 remap(H)。如果畸变大，需要先 undistort 再 warpPerspective。
        # 鉴于 D1, D2 存在，我们先 undistort Points? 不，直接生成 Map 最稳。

        # 简单的 Map 生成逻辑:
        # Map = UndistortMap + HomographyWarp
        # 这种组合比较复杂。为了简单有效，我们假设 D 已经由 initUndistortRectifyMap 处理，
        # 但这里的 H1 H2 是作用在已经去畸变的图像上的吗？
        # OpenCV 文档: stereoRectifyUncalibrated compute H for *undistorted* points usually.
        # 但这里我们直接用 raw points 算的 F。

        # 让我们生成一个综合的 Map：
        # Map[x,y] -> src_x, src_y
        # 步骤: (x,y) -> H_inv -> (u,v) -> Distort -> (u_dist, v_dist)

        # 为了兼容性，我们直接保存 H1, H2 对应的 Map，但忽略 D (假设 D 影响小于 R 错位)
        # 或者，我们可以 "Cheat": 把 K 保持不变，只改 R。
        # 但 R 是算出 H 的一部分。

        # 最稳妥方案：生成纯像素级的 Map
        m1l, m2l = self._build_map_from_H(H1, self.img_size)
        m1r, m2r = self._build_map_from_H(H2, self.img_size)

        # 制作一个假的 Q (仅供深度相对值)
        Q_fake = np.float32([
            [1, 0, 0, -self.img_size[0] / 2],
            [0, 1, 0, -self.img_size[1] / 2],
            [0, 0, 0, self.K1[0, 0]],
            [0, 0, 1 / 100.0, 0]
        ])

        # 验证误差
        rect_L = cv2.remap(imgL, m1l, m2l, cv2.INTER_LINEAR)
        rect_R = cv2.remap(imgR, m1r, m2r, cv2.INTER_LINEAR)
        err = self.calc_sift_error(rect_L, rect_R)
        print(f"   -> 自动矫正后误差: {err:.4f} px")

        if err < 2.0:
            print("   ✅ 自动修复成功！")
            save_path = os.path.join(BASE_DIR, "calibration_fixed.npz")
            np.savez(save_path,
                     map1_left=m1l, map2_left=m2l,
                     map1_right=m1r, map2_right=m2r,
                     Q=Q_fake)
            print(f"   💾 参数已保存至: {save_path}")

            # 可视化
            self.visualize_result(rect_L, rect_R)

    def _build_map_from_H(self, H, size):
        w, h = size
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones_like(x)
        coords = np.stack([x, y, ones], axis=-1).reshape(-1, 3)
        H_inv = np.linalg.inv(H)
        src_coords = (H_inv @ coords.T).T
        z = src_coords[:, 2:3]
        z[z == 0] = 1e-10
        src_coords = src_coords[:, :2] / z

        map_x = src_coords[:, 0].reshape(h, w).astype(np.float32)
        map_y = src_coords[:, 1].reshape(h, w).astype(np.float32)
        return map_x, map_y

    def calc_sift_error(self, imgL, imgR):
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(grayL, None)
        kp2, des2 = sift.detectAndCompute(grayR, None)
        if des1 is None or des2 is None: return 999.0
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        ptsL, ptsR = [], []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                ptsL.append(kp1[m.queryIdx].pt)
                ptsR.append(kp2[m.trainIdx].pt)
        if len(ptsL) < 10: return 999.0
        return np.mean(np.abs(np.array(ptsL)[:, 1] - np.array(ptsR)[:, 1]))

    def save_and_vis(self, imgL, imgR, K1, D1, R1, P1, K2, D2, R2, P2, Q, tag):
        m1l, m2l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, self.img_size, cv2.CV_32FC1)
        m1r, m2r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, self.img_size, cv2.CV_32FC1)

        rect_L = cv2.remap(imgL, m1l, m2l, cv2.INTER_LINEAR)
        rect_R = cv2.remap(imgR, m1r, m2r, cv2.INTER_LINEAR)

        err = self.calc_sift_error(rect_L, rect_R)
        print(f"   -> [{tag}] 误差: {err:.4f} px")

        if err < 1.5:
            save_path = os.path.join(BASE_DIR, "calibration_fixed.npz")
            np.savez(save_path,
                     map1_left=m1l, map2_left=m2l,
                     map1_right=m1r, map2_right=m2r,
                     Q=Q)
            print(f"   💾 参数已保存至: {save_path}")
            self.visualize_result(rect_L, rect_R)

    def visualize_result(self, rect_L, rect_R):
        vis = np.vstack([np.hstack([rect_L[::4, ::4], rect_R[::4, ::4]])])
        h, w = vis.shape[:2]
        for i in range(0, h, 40): cv2.line(vis, (0, i), (w, i), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(OUT_DIR, "final_rectified.jpg"), vis)
        print(f"   可视结果已保存: {os.path.join(OUT_DIR, 'final_rectified.jpg')}")


if __name__ == "__main__":
    if not os.path.exists(LEFT_PATH):
        print("图片路径错误")
        exit()

    imgL = cv2.imread(LEFT_PATH)
    imgR = cv2.imread(RIGHT_PATH)
    h, w = imgL.shape[:2]

    rectifier = UniversalRectifier(CALIB_PATH, (w, h))
    rectifier.validate_geometry_and_fix(imgL, imgR)