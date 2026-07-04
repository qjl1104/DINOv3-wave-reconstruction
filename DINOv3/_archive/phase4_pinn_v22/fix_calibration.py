import cv2
import numpy as np
import os
from scipy.optimize import minimize

# ----------------------------------------------------------------------------------
# 配置区域
# ----------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(BASE_DIR, "calibration_paper_params.npz")
LEFT_PATH = r"D:\Research\wave_reconstruction_project\data\left_images\left0001.bmp"
RIGHT_PATH = r"D:\Research\wave_reconstruction_project\data\right_images\right0001.bmp"
OUT_DIR = os.path.join(BASE_DIR, "rectified_output")
os.makedirs(OUT_DIR, exist_ok=True)


class CalibrationFixer:
    def __init__(self, calib_path, imgL, imgR):
        self.calib_path = calib_path
        self.imgL = imgL
        self.imgR = imgR
        self.h, self.w = imgL.shape[:2]
        self.load_params()

    def load_params(self):
        data = np.load(self.calib_path)
        # 兼容性读取
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

        print(f"[Init] 原始光心 (cx, cy):")
        print(f"  Left : ({self.K1[0, 2]:.2f}, {self.K1[1, 2]:.2f})")
        print(f"  Right: ({self.K2[0, 2]:.2f}, {self.K2[1, 2]:.2f})")
        print(f"  Image Center: ({self.w / 2}, {self.h / 2})")

    def get_matches(self):
        print("[Process] 提取 SIFT 特征用于对齐...")
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.imgL, None)
        kp2, des2 = sift.detectAndCompute(self.imgR, None)

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)

        pts1, pts2 = [], []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)

        self.pts1 = np.array(pts1)
        self.pts2 = np.array(pts2)
        print(f"  -> 使用 {len(self.pts1)} 对特征点进行优化")
        return len(self.pts1) > 20

    def objective_function(self, shift_params):
        """
        优化目标函数：最小化对极误差 (Pixel Units)
        """
        dL_x, dL_y, dR_x, dR_y = shift_params

        # 构造临时 K
        K1_new = self.K1.copy()
        K2_new = self.K2.copy()
        K1_new[0, 2] += dL_x
        K1_new[1, 2] += dL_y
        K2_new[0, 2] += dR_x
        K2_new[1, 2] += dR_y

        # 1. Undistort Points (使用新的 K)
        # 注意：undistortPoints 返回的是归一化平面坐标 (x, y)
        pts1_norm = cv2.undistortPoints(self.pts1.reshape(-1, 1, 2), K1_new, self.D1)
        pts2_norm = cv2.undistortPoints(self.pts2.reshape(-1, 1, 2), K2_new, self.D2)

        # 2. 计算 Essential Matrix E
        T_x = np.array([
            [0, -self.T[2], self.T[1]],
            [self.T[2], 0, -self.T[0]],
            [-self.T[1], self.T[0], 0]
        ])
        E = T_x @ self.R

        # 3. Sampson Error (Pixels)
        # 我们需要将归一化误差转回像素误差，以便优化器能感知到梯度
        # 归一化误差 distance ~ pixel_error / focal_length
        # 所以 pixel_error ~ distance * focal_length

        N = len(pts1_norm)
        x1 = np.hstack([pts1_norm.reshape(N, 2), np.ones((N, 1))])
        x2 = np.hstack([pts2_norm.reshape(N, 2), np.ones((N, 1))])

        # l2 = E * x1 (归一化平面上的极线)
        l2 = (E @ x1.T).T

        # 归一化距离 d = |x2^T * l2| / |l2|
        numer = np.abs(np.sum(x2 * l2, axis=1))
        denom = np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2)
        dist_norm = numer / denom

        # [Fix] 乘以焦距，转换为像素误差
        f_mean = (self.K1[0, 0] + self.K2[0, 0]) / 2.0
        error_pixel = np.mean(dist_norm) * f_mean

        return error_pixel

    def optimize(self):
        if not self.get_matches():
            return

        print("[Optimize] 开始搜索最佳光心偏移 (Pixel Error Mode)...")
        # 初始猜测：假设没有偏移
        x0 = [0.0, 0.0, 0.0, 0.0]

        # 使用 Powell 算法，这种算法对坐标下降很有效，不需要导数
        # 之前 Nelder-Mead 因为误差数值太小而早停
        res = minimize(self.objective_function, x0, method='Powell', tol=0.1)

        print(f"\n[Result] 优化完成!")
        print(f"  初始误差: {self.objective_function(x0):.4f} px")
        print(f"  优化误差: {res.fun:.4f} px")
        print(f"  偏移量 (dL_x, dL_y, dR_x, dR_y): {np.round(res.x, 2)}")

        self.apply_fix_and_save(res.x)

    def apply_fix_and_save(self, shifts):
        dL_x, dL_y, dR_x, dR_y = shifts

        # 修正 K
        self.K1[0, 2] += dL_x
        self.K1[1, 2] += dL_y
        self.K2[0, 2] += dR_x
        self.K2[1, 2] += dR_y

        print("\n[Step 3] 使用修正后的 K 重新生成 Map...")
        # 重新计算 Map
        size = (self.w, self.h)
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
        )

        m1l, m2l = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, size, cv2.CV_32FC1)
        m1r, m2r = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2, size, cv2.CV_32FC1)

        # 验证最终矫正效果 (remap后测Y差)
        rectL = cv2.remap(self.imgL, m1l, m2l, cv2.INTER_LINEAR)
        rectR = cv2.remap(self.imgR, m1r, m2r, cv2.INTER_LINEAR)

        err = self.calc_y_error(rectL, rectR)
        print(f"  -> 最终矫正后平均行对齐误差: {err:.4f} px")

        if err < 1.5:
            print("✅ 成功！已自动修复裁剪偏移。")
            save_path = os.path.join(BASE_DIR, "calibration_fixed.npz")
            np.savez(save_path,
                     map1_left=m1l, map2_left=m2l,
                     map1_right=m1r, map2_right=m2r,
                     Q=Q)  # 注意：这里Q可能也需要微调，但通常影响不大
            print(f"💾 已保存修正参数至: {save_path}")

            vis = np.vstack([np.hstack([rectL[::4, ::4], rectR[::4, ::4]])])
            h, w = vis.shape[:2]
            for i in range(0, h, 40): cv2.line(vis, (0, i), (w, i), (0, 255, 0), 1)
            cv2.imwrite(os.path.join(OUT_DIR, "fixed_result.jpg"), vis)  # 保存结果
            cv2.imshow("Fixed Result", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ 依然无法完全修复，可能不仅仅是偏移问题 (或者 R/T 真的变了)。")

    def calc_y_error(self, imgL, imgR):
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
        if not ptsL: return 999.0
        return np.mean(np.abs(np.array(ptsL)[:, 1] - np.array(ptsR)[:, 1]))


if __name__ == "__main__":
    imgL = cv2.imread(LEFT_PATH)
    imgR = cv2.imread(RIGHT_PATH)
    fixer = CalibrationFixer(CALIB_PATH, imgL, imgR)
    fixer.optimize()