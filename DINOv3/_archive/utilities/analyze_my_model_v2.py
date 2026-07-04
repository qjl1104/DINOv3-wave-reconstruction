# evaluate_photometric_quality_v2.py
# 描述:
# 这是针对稀疏、高对比度图像（如白点/黑背景）优化的评估脚本。
#
# 核心修改：
# 1. 不再使用原始像素计算SAD，因为该指标对稀疏数据极其“脆弱”，
#    1个像素的误差就会导致SAD值剧烈升高，从而产生误导性结果。
# 2. 引入 "Blurred SAD" (模糊SAD) 作为新指标：
#    在比较图像块之前，先对左右校正图像应用高斯模糊(Gaussian Blur)。
#    这使得SAD指标对微小的空间位移更加宽容，能更准确地衡量
#    匹配点之间的“邻近程度”，而不是“精确重叠”。
#
# 这个脚本的结果更能反映模型真实的几何匹配质量。

import os
import sys
import glob
import time
import json
import dataclasses

import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

try:
    from sparse_reconstructor_1013_gemini import Config, SparseMatchingStereoModel
except ImportError:
    print("错误: 无法导入 'sparse_reconstructor_1013_gemini.py'。")
    sys.exit(1)


def find_latest_run_dir():
    """自动查找并返回最新的训练运行目录"""
    # 尝试从 common_paths.py 或类似文件导入项目根目录
    try:
        # 假设 sparse_reconstructor_1013_gemini.py 中定义了 PROJECT_ROOT
        from sparse_reconstructor_1013_gemini import PROJECT_ROOT
        print(f"从 'sparse_reconstructor_1013_gemini.py' 加载 PROJECT_ROOT: {PROJECT_ROOT}")
    except ImportError:
        print("[Warning] 无法从 'sparse_reconstructor_1013_gemini.py' 导入 PROJECT_ROOT。")
        print("[Warning] 将使用当前脚本所在目录作为 PROJECT_ROOT。")
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    runs_base_dir = os.path.join(PROJECT_ROOT, "training_runs_sparse")

    if not os.path.exists(runs_base_dir):
        # 如果在导入的 PROJECT_ROOT 中找不到，尝试回退到本地目录
        local_runs_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_runs_sparse")
        if os.path.exists(local_runs_base_dir):
            runs_base_dir = local_runs_base_dir
            print(f"回退到本地 'training_runs_sparse' 目录: {runs_base_dir}")
        else:
            sys.exit(f"错误: 基础运行目录不存在 '{runs_base_dir}' 或 '{local_runs_base_dir}'")

    try:
        latest_run_name = \
        sorted([d for d in os.listdir(runs_base_dir) if os.path.isdir(os.path.join(runs_base_dir, d))])[-1]
    except IndexError:
        sys.exit(f"错误: 在 '{runs_base_dir}' 中没有找到任何训练运行目录。")

    latest_run_path = os.path.join(runs_base_dir, latest_run_name)
    print(f"自动检测到最新的运行目录: {latest_run_path}")
    return latest_run_path


class PhotometricQualityAnalyzer:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.output_dir = os.path.join(self.run_dir, "evaluation_results")
        os.makedirs(self.output_dir, exist_ok=True)  # 确保目录存在
        self.cfg = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.calib = self._load_calibration()
        self.model = SparseMatchingStereoModel(self.cfg).to(self.device)
        model_path = os.path.join(self.run_dir, "checkpoints", "best_model_sparse.pth")

        if not os.path.exists(model_path):
            sys.exit(f"错误: 找不到模型文件 '{model_path}'")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.left_images = sorted(glob.glob(os.path.join(self.cfg.LEFT_IMAGE_DIR, "*.*")))
        print(f"分析器初始化完毕，共找到 {len(self.left_images)} 张左图。")

    def _load_config(self) -> Config:
        log_file = os.path.join(self.run_dir, "logs", "training_log.json")
        if not os.path.exists(log_file):
            sys.exit(f"错误: 找不到日志文件 '{log_file}'，无法加载配置。")
        with open(log_file, 'r') as f:
            config_dict = json.load(f)['config']
        config_fields = {f.name for f in dataclasses.fields(Config)}
        return Config(**{k: v for k, v in config_dict.items() if k in config_fields})

    def _load_calibration(self):
        try:
            return np.load(self.cfg.CALIBRATION_FILE)
        except Exception as e:
            sys.exit(f"加载标定文件失败: {e}")

    def get_matches_and_rectified_images(self, img_path_l, img_path_r):
        """获取模型匹配结果和用于评估的、校正后的全尺寸图像"""
        img_l_raw, img_r_raw = cv2.imread(img_path_l, 0), cv2.imread(img_path_r, 0)
        if img_l_raw is None or img_r_raw is None:
            print(f"警告: 无法读取图像 {img_path_l} 或 {img_path_r}")
            return None, None, None, None

        img_l_rect = cv2.remap(img_l_raw, self.calib['map1_left'], self.calib['map2_left'], cv2.INTER_LINEAR)
        img_r_rect = cv2.remap(img_r_raw, self.calib['map1_right'], self.calib['map2_right'], cv2.INTER_LINEAR)

        x_l, y_l, w_l, h_l = self.calib['roi_left'].astype(int)
        img_l_roi = img_l_rect[y_l:y_l + h_l, x_l:x_l + w_l]

        x_r, y_r, w_r, h_r = self.calib['roi_right'].astype(int)
        img_r_roi = img_r_rect[y_r:y_r + h_r, x_r:x_r + w_r]

        img_l_resized = cv2.resize(img_l_roi, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
        img_r_resized = cv2.resize(img_r_roi, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))

        left_gray_tensor = torch.from_numpy(img_l_resized).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
        right_gray_tensor = torch.from_numpy(img_r_resized).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0

        # 确保为 DINOv3 准备 RGB (3-channel) 图像
        left_rgb_tensor = torch.from_numpy(
            cv2.cvtColor(img_l_resized, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float().unsqueeze(0).to(
            self.device) / 255.0
        right_rgb_tensor = torch.from_numpy(
            cv2.cvtColor(img_r_resized, cv2.COLOR_GRAY2BGR).transpose(2, 0, 1)).float().unsqueeze(0).to(
            self.device) / 255.0

        # 创建一个全1的掩码，因为我们的关键点检测器会自己找亮点
        mask_tensor = torch.ones_like(left_gray_tensor)

        with torch.no_grad():
            outputs = self.model(left_gray_tensor, right_gray_tensor, left_rgb_tensor, right_rgb_tensor,
                                 mask_tensor)

        kp_l_model, scores_l, disp_model = outputs['keypoints_left'][0].cpu().numpy(), outputs['scores_left'][
            0].cpu().numpy(), outputs['disparity'][0].cpu().numpy()

        valid_mask = scores_l > 0.1
        kp_l_model, disp_model = kp_l_model[valid_mask], disp_model[valid_mask]

        if len(kp_l_model) == 0: return np.array([]), np.array([]), img_l_rect, img_r_rect

        scale_x_roi, scale_y_roi = w_l / self.cfg.IMAGE_WIDTH, h_l / self.cfg.IMAGE_HEIGHT
        kp_l_roi = kp_l_model * [scale_x_roi, scale_y_roi]
        disp_roi = disp_model * scale_x_roi
        kp_l_full = kp_l_roi + [x_l, y_l]
        kp_r_full = kp_l_full.copy()
        kp_r_full[:, 0] -= disp_roi

        return kp_l_full, kp_r_full, img_l_rect, img_r_rect

    def analyze_single_frame(self, image_index):
        if image_index >= len(self.left_images):
            print(f"警告: 索引 {image_index} 超出图像列表范围 (总数 {len(self.left_images)})。")
            return None

        left_path = self.left_images[image_index]
        right_path = os.path.join(self.cfg.RIGHT_IMAGE_DIR, 'right' + os.path.basename(left_path)[4:])

        start_time = time.perf_counter()
        pts_l, pts_r, img_l_rect, img_r_rect = self.get_matches_and_rectified_images(left_path, right_path)
        duration = time.perf_counter() - start_time

        if pts_l is None: return None

        num_matches = len(pts_l)
        if num_matches == 0:
            return {'Matches': 0, 'Blurred SAD': np.nan, 'Photometric Inlier Ratio (%)': 0,
                    'FPS': 1 / (duration + 1e-9)}

        # --- 核心修改 ---
        # 1. 定义模糊核大小。(patch_size * 2 + 1) 是一个好选择，确保模糊范围大于块。
        patch_size = 7
        blur_ksize = (patch_size * 2 + 1, patch_size * 2 + 1)  # e.g., (15, 15)

        # 2. 对全尺寸校正图像应用高斯模糊
        # sigmaX=0 会让cv2自动根据ksize计算标准差
        img_l_blurred = cv2.GaussianBlur(img_l_rect, blur_ksize, sigmaX=0)
        img_r_blurred = cv2.GaussianBlur(img_r_rect, blur_ksize, sigmaX=0)
        # --- 修改结束 ---

        half_patch = patch_size // 2
        photometric_errors = []
        h, w = img_l_rect.shape

        for pt_l, pt_r in zip(pts_l, pts_r):
            x_l, y_l = int(round(pt_l[0])), int(round(pt_l[1]))
            x_r, y_r = int(round(pt_r[0])), int(round(pt_r[1]))

            if (y_l - half_patch < 0 or y_l + half_patch + 1 > h or x_l - half_patch < 0 or x_l + half_patch + 1 > w or
                    y_r - half_patch < 0 or y_r + half_patch + 1 > h or x_r - half_patch < 0 or x_r + half_patch + 1 > w):
                photometric_errors.append(np.nan)
                continue

            # 3. 从 *模糊后* 的图像中提取块
            patch_l = img_l_blurred[y_l - half_patch: y_l + half_patch + 1, x_l - half_patch: x_l + half_patch + 1]
            patch_r = img_r_blurred[y_r - half_patch: y_r + half_patch + 1, x_r - half_patch: x_r + half_patch + 1]

            sad = np.sum(np.abs(patch_l.astype(float) - patch_r.astype(float)))
            photometric_errors.append(sad)

        photometric_errors = np.array(photometric_errors)
        mean_photometric_error = np.nanmean(photometric_errors)

        # 计算光度一致内点率
        # 定义内点：光度误差小于 均值+1倍标准差
        # 这个动态阈值现在在 "Blurred SAD" 上计算，会更有意义
        with np.errstate(invalid='ignore'):  # 忽略nanmean/nanstd的RuntimeWarning
            photo_error_thresh = np.nanmean(photometric_errors) + np.nanstd(photometric_errors)

        if np.isnan(photo_error_thresh):
            inlier_ratio = 0.0  # 如果所有都是nan，则内点率为0
        else:
            inliers = photometric_errors < photo_error_thresh
            inlier_ratio = np.sum(inliers) / num_matches * 100

        return {
            'Matches': num_matches,
            'Blurred SAD': mean_photometric_error,  # 重命名指标
            'Photometric Inlier Ratio (%)': inlier_ratio,
            'FPS': 1 / duration
        }


def main():
    latest_run_dir = find_latest_run_dir()
    if not latest_run_dir:
        print("未找到运行目录，退出。")
        return

    analyzer = PhotometricQualityAnalyzer(run_dir=latest_run_dir)

    IMAGE_INDICES_TO_EVALUATE = [100, 250, 400, 600, 800]

    all_results = [analyzer.analyze_single_frame(idx) for idx in tqdm(IMAGE_INDICES_TO_EVALUATE, desc="评估帧")]
    all_results = [r for r in all_results if r is not None]

    if all_results:
        df = pd.DataFrame(all_results)
        summary = df.mean().to_frame().T.round(3)

        print("\n\n" + "=" * 60)
        print("    你的模型 (DINOv3+Attention) 优化版光度质量定量分析")
        print("           (使用 'Blurred SAD' 评估稀疏数据)")
        print("=" * 60)
        print("\n--- 光度匹配质量 (多帧平均值) ---\n")
        print(summary.to_string(index=False))

        output_path = os.path.join(analyzer.output_dir, "my_model_blurred_sad_results.csv")
        summary.to_csv(output_path, index=False)
        print(f"\n\n定量分析结果已保存至: {output_path}")
    else:
        print("评估完成，但没有收集到任何有效结果。")


if __name__ == "__main__":
    main()
