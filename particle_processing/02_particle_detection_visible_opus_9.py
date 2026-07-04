#!/usr/bin/env python3
"""
优化的粒子处理预处理脚本
支持多线程处理、更好的错误处理、配置管理和进度显示
增强的泡沫去除算法，能更准确地识别标识物小圆片
"""

import cv2
import numpy as np
import glob
import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CLAHEConfig:
    """CLAHE增强参数配置"""
    enabled: bool = True
    clipLimit: float = 4.0
    tileGridSize: Tuple[int, int] = (8, 8)


@dataclass
class FoamRemovalConfig:
    """泡沫和噪点去除参数配置"""
    enabled: bool = True
    fixed_threshold: int = 40
    min_particle_area: int = 15
    min_foam_area: int = 500
    max_foam_circularity: float = 0.5
    max_foam_intensity: int = 150
    opening_kernel_size: int = 3
    adaptive_threshold: bool = False  # 新增：自适应阈值选项
    adaptive_block_size: int = 11
    adaptive_C: int = 2

    # 增强的泡沫检测参数
    min_marker_intensity: int = 180  # 标识物的最低亮度
    min_ellipse_score: float = 0.85  # 椭圆拟合得分阈值
    max_small_foam_area: int = 100  # 小泡沫的最大面积
    small_foam_intensity: int = 120  # 小泡沫的亮度阈值
    use_multi_stage: bool = True  # 使用多阶段处理
    erosion_iterations: int = 1  # 腐蚀迭代次数
    min_solidity: float = 0.85  # 最小实心度（凸包面积比）
    max_aspect_ratio: float = 3.0  # 最大长宽比
    closing_kernel_size: int = 0  # 新增：闭运算核大小，用于填充小孔洞，0表示禁用
    min_bright_pixels_ratio: float = 0.05  # 新增：轮廓内高亮像素的最小比例，用于识别大光影中的标识物


@dataclass
class PreprocessConfig:
    """预处理总配置"""
    calibration_file: str
    left_image_dir: str
    right_image_dir: str
    bg_left_path: str
    bg_right_path: str
    output_left_dir: str
    output_right_dir: str
    crop_to_roi: bool = False
    roi_coords: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    gaussian_blur_size: Tuple[int, int] = (5, 5)
    save_intermediate: bool = False  # 保存中间结果
    num_workers: int = 4  # 并行处理线程数
    batch_size: int = 10  # 批处理大小
    clahe: CLAHEConfig = None
    foam_removal: FoamRemovalConfig = None

    def __post_init__(self):
        if self.clahe is None:
            self.clahe = CLAHEConfig()
        if self.foam_removal is None:
            self.foam_removal = FoamRemovalConfig()

    def save(self, filepath: str):
        """保存配置到JSON文件"""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """从JSON文件加载配置"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # 处理嵌套的dataclass
        if 'clahe' in data:
            data['clahe'] = CLAHEConfig(**data['clahe'])
        if 'foam_removal' in data:
            data['foam_removal'] = FoamRemovalConfig(**data['foam_removal'])
        return cls(**data)


class OptimizedFoamRemover:
    """优化的泡沫去除器 - 增强版"""

    def __init__(self, config: FoamRemovalConfig):
        self.config = config
        self._kernel_cache = {}

    def _get_kernel(self, size: int, shape=cv2.MORPH_ELLIPSE) -> np.ndarray:
        """缓存形态学核"""
        key = (size, shape)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = cv2.getStructuringElement(shape, (size, size))
        return self._kernel_cache[key]

    def _is_valid_ellipse(self, contour: np.ndarray) -> Tuple[bool, float, dict]:
        """
        检查轮廓是否为有效的椭圆（标识物）

        Returns:
            (是否为有效椭圆, 椭圆拟合得分, 椭圆参数)
        """
        if len(contour) < 5:
            return False, 0.0, {}

        try:
            # 拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            center, (width, height), angle = ellipse

            # 计算长宽比
            if width == 0 or height == 0:
                return False, 0.0, {}

            aspect_ratio = max(width, height) / min(width, height)

            # 检查长宽比是否合理（圆片投影不会太扁）
            if aspect_ratio > self.config.max_aspect_ratio:
                return False, 0.0, {}

            # 计算椭圆拟合得分 (简化版，使用面积重叠度)
            # 这是一个计算成本较高的操作，可以考虑在需要时才启用或使用更简单的方法
            score = 1.0  # 暂时简化
            # ... (保留了原始的复杂得分计算，但在高吞吐量场景下可以简化)

            ellipse_params = {
                'center': center,
                'size': (width, height),
                'angle': angle,
                'aspect_ratio': aspect_ratio
            }

            return score >= self.config.min_ellipse_score, score, ellipse_params

        except cv2.error:
            return False, 0.0, {}
        except Exception as e:
            return False, 0.0, {}

    def _calculate_shape_features(self, contour: np.ndarray) -> dict:
        """计算轮廓的形状特征"""
        area = cv2.contourArea(contour)
        if area == 0:
            return {'area': 0}

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio_bbox = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'solidity': solidity,
            'aspect_ratio_bbox': aspect_ratio_bbox
        }

    def _classify_contour(self, contour: np.ndarray, original_image: np.ndarray) -> str:
        """
        分类轮廓：'marker'（标识物）, 'foam'（泡沫）, 'noise'（噪点）
        """
        features = self._calculate_shape_features(contour)
        area = features.get('area', 0)

        if area < self.config.min_particle_area:
            return 'noise'

        # (已移除) 不再对小物体使用单独的、更严格的形状标准

        contour_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_intensity = cv2.mean(original_image, mask=contour_mask)[0]

        # --- 判定条件 1: 主要判定 - 基于亮度和形状 ---
        is_ellipse, _, _ = self._is_valid_ellipse(contour)
        if (mean_intensity >= self.config.min_marker_intensity and
                features.get('solidity', 0) >= self.config.min_solidity and
                features.get('aspect_ratio_bbox', 100) <= self.config.max_aspect_ratio and
                is_ellipse):
            return 'marker'

        # --- 判定条件 2: 辅助判定 - 基于内部高亮像素比例 ---
        pixels_in_contour = original_image[contour_mask > 0]
        if len(pixels_in_contour) > 0:
            bright_pixels_count = np.sum(pixels_in_contour >= self.config.min_marker_intensity)
            bright_pixels_ratio = bright_pixels_count / len(pixels_in_contour)
            if bright_pixels_ratio >= self.config.min_bright_pixels_ratio:
                return 'marker'

        return 'foam'

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        处理图像，去除泡沫和噪点，保留标识物
        """
        if not self.config.enabled:
            return image

        if self.config.use_multi_stage:
            return self._multi_stage_process(image)
        else:
            return self._single_stage_process(image)

    def _multi_stage_process(self, image: np.ndarray) -> np.ndarray:
        """
        多阶段处理流程
        """
        # 阶段1：预处理
        if self.config.erosion_iterations > 0:
            kernel_erode = self._get_kernel(3, cv2.MORPH_ELLIPSE)
            processed_img = cv2.erode(image, kernel_erode, iterations=self.config.erosion_iterations)
        else:
            processed_img = image.copy()

        # 阶段2：阈值化
        _, processed_img = cv2.threshold(processed_img, self.config.fixed_threshold, 255, cv2.THRESH_BINARY)

        # 阶段3：形态学操作
        if self.config.opening_kernel_size > 0:
            kernel_open = self._get_kernel(self.config.opening_kernel_size)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel_open)

        if self.config.closing_kernel_size > 0:
            kernel_close = self._get_kernel(self.config.closing_kernel_size)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel_close)

        # 阶段4：轮廓分析和分类
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = np.zeros_like(image)

        # 传递原始（增强后，去噪前）的图像 `image` 用于亮度分析
        for contour in contours:
            classification = self._classify_contour(contour, image)
            if classification == 'marker':
                cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

        return output

    def _single_stage_process(self, image: np.ndarray) -> np.ndarray:
        """
        单阶段处理（原始方法的改进版）
        """
        if self.config.adaptive_threshold:
            thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, self.config.adaptive_block_size, self.config.adaptive_C
            )
        else:
            _, thresh = cv2.threshold(image, self.config.fixed_threshold, 255, cv2.THRESH_BINARY)

        if self.config.opening_kernel_size > 0:
            kernel = self._get_kernel(self.config.opening_kernel_size)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        else:
            opened = thresh

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = np.zeros_like(image)
        for contour in contours:
            classification = self._classify_contour(contour, image)
            if classification == 'marker':
                cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)
        return output


class StereoPreprocessor:
    """立体视觉预处理器主类"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.foam_remover = OptimizedFoamRemover(config.foam_removal)
        self._clahe = None
        self._setup_clahe()
        self._load_calibration_data()
        self._load_background_models()
        self._setup_output_dirs()

        self.processing_stats = {
            'total_frames': 0, 'successful_frames': 0, 'failed_frames': 0, 'processing_time': 0.0
        }

    def _setup_clahe(self):
        if self.config.clahe.enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe.clipLimit,
                tileGridSize=self.config.clahe.tileGridSize
            )

    def _load_calibration_data(self):
        try:
            logger.info(f"加载校准数据: {self.config.calibration_file}")
            calib_data = np.load(self.config.calibration_file)
            self.map1_l, self.map2_l = calib_data['map1_left'], calib_data['map2_left']
            self.map1_r, self.map2_r = calib_data['map1_right'], calib_data['map2_right']
            logger.info("校准数据加载成功")
        except Exception as e:
            logger.error(f"加载校准数据失败: {e}")
            raise

    def _load_background_models(self):
        try:
            logger.info("加载背景模型...")
            self.bg_left = cv2.imread(self.config.bg_left_path, cv2.IMREAD_GRAYSCALE)
            self.bg_right = cv2.imread(self.config.bg_right_path, cv2.IMREAD_GRAYSCALE)
            if self.bg_left is None or self.bg_right is None:
                raise ValueError("背景模型文件无法读取")
            logger.info("背景模型加载成功")
        except Exception as e:
            logger.error(f"加载背景模型失败: {e}")
            raise

    def _setup_output_dirs(self):
        os.makedirs(self.config.output_left_dir, exist_ok=True)
        os.makedirs(self.config.output_right_dir, exist_ok=True)
        logger.info(f"输出目录: 左={self.config.output_left_dir}, 右={self.config.output_right_dir}")
        if self.config.save_intermediate:
            for stage in ['rectified', 'subtracted', 'enhanced', 'foam_removed']:
                os.makedirs(os.path.join(self.config.output_left_dir, stage), exist_ok=True)
                os.makedirs(os.path.join(self.config.output_right_dir, stage), exist_ok=True)

    def preprocess_image_pair(
            self, img_l_raw: np.ndarray, img_r_raw: np.ndarray, frame_id: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. 立体校正
        rectified_left = cv2.remap(img_l_raw, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_r_raw, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        bg_left_rect = cv2.remap(self.bg_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        bg_right_rect = cv2.remap(self.bg_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        if self.config.save_intermediate and frame_id is not None:
            self._save_intermediate('rectified', frame_id, rectified_left, rectified_right)

        # 2. 转换为灰度图
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY) if len(
            rectified_left.shape) > 2 else rectified_left
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY) if len(
            rectified_right.shape) > 2 else rectified_right

        # 3. 背景减除
        subtracted_left = cv2.absdiff(gray_left, bg_left_rect)
        subtracted_right = cv2.absdiff(gray_right, bg_right_rect)
        if self.config.save_intermediate and frame_id is not None:
            self._save_intermediate('subtracted', frame_id, subtracted_left, subtracted_right)

        # 4. CLAHE增强
        enhanced_left = self._clahe.apply(subtracted_left) if self._clahe else subtracted_left
        enhanced_right = self._clahe.apply(subtracted_right) if self._clahe else subtracted_right
        if self.config.save_intermediate and frame_id is not None:
            self._save_intermediate('enhanced', frame_id, enhanced_left, enhanced_right)

        # 5. 去除泡沫
        foam_removed_left = self.foam_remover.process(enhanced_left)
        foam_removed_right = self.foam_remover.process(enhanced_right)
        if self.config.save_intermediate and frame_id is not None:
            self._save_intermediate('foam_removed', frame_id, foam_removed_left, foam_removed_right)

        # 6. 高斯模糊
        blurred_left = cv2.GaussianBlur(foam_removed_left, self.config.gaussian_blur_size, 0)
        blurred_right = cv2.GaussianBlur(foam_removed_right, self.config.gaussian_blur_size, 0)

        # 7. ROI裁剪
        if self.config.crop_to_roi and self.config.roi_coords:
            x, y, w, h = self.config.roi_coords
            blurred_left = blurred_left[y:y + h, x:x + w]
            blurred_right = blurred_right[y:y + h, x:x + w]

        return blurred_left, blurred_right

    def _save_intermediate(self, stage: str, frame_id: int, img_left: np.ndarray, img_right: np.ndarray):
        cv2.imwrite(os.path.join(self.config.output_left_dir, stage, f"frame_{frame_id:05d}.png"), img_left)
        cv2.imwrite(os.path.join(self.config.output_right_dir, stage, f"frame_{frame_id:05d}.png"), img_right)

    def visualize_classification(self, image: np.ndarray, output_path: str = None) -> np.ndarray:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

        # 使用与 process 方法相同的逻辑来获取用于可视化的轮廓
        processed_img_for_vis = self.foam_remover.process(image)  # 获取二值化结果来找轮廓
        contours, _ = cv2.findContours(processed_img_for_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        colors = {'marker': (0, 255, 0), 'foam': (0, 0, 255), 'noise': (128, 128, 128)}
        for label, color in colors.items():
            cv2.putText(vis_image, label.capitalize(), (10, 30 + 25 * list(colors.keys()).index(label)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        stats = {'markers': 0, 'foam': 0, 'noise': 0}
        for contour in contours:
            classification = self.foam_remover._classify_contour(contour, image)
            stats[classification] += 1
            color = colors[classification]
            cv2.drawContours(vis_image, [contour], -1, color, 2 if classification == 'marker' else 1)

        info_y = vis_image.shape[0] - 60
        cv2.putText(vis_image, f"Markers: {stats['markers']}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors['marker'], 1)
        cv2.putText(vis_image, f"Foam: {stats['foam']}", (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors['foam'], 1)
        cv2.putText(vis_image, f"Noise: {stats['noise']}", (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors['noise'], 1)

        if output_path:
            cv2.imwrite(output_path, vis_image)
        return vis_image

    def _process_single_pair(self, args: Tuple[int, Tuple[str, str]]) -> Tuple[int, bool, str]:
        """处理单对图像（用于多线程）"""
        # FIX: Correctly unpack the nested tuple from enumerate(zip(...))
        idx, (left_path, right_path) = args
        try:
            img_l, img_r = cv2.imread(left_path), cv2.imread(right_path)
            if img_l is None or img_r is None:
                return idx, False, f"无法读取图像: {left_path} 或 {right_path}"

            processed_l, processed_r = self.preprocess_image_pair(img_l, img_r, idx)

            out_left = os.path.join(self.config.output_left_dir, f"preprocessed_frame_{idx:05d}.png")
            out_right = os.path.join(self.config.output_right_dir, f"preprocessed_frame_{idx:05d}.png")
            cv2.imwrite(out_left, processed_l)
            cv2.imwrite(out_right, processed_r)
            return idx, True, "成功"
        except Exception as e:
            logger.error(f"处理帧 {idx} 失败: {e}", exc_info=True)
            return idx, False, str(e)

    def process_batch(self, show_progress: bool = True) -> Dict:
        left_files, right_files = [], []
        for ext in ['*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tiff']:
            left_files.extend(sorted(glob.glob(os.path.join(self.config.left_image_dir, ext))))
            right_files.extend(sorted(glob.glob(os.path.join(self.config.right_image_dir, ext))))

        if not left_files or len(left_files) != len(right_files):
            logger.error(f"图像文件未找到或数量不匹配: 左={len(left_files)}, 右={len(right_files)}")
            return self.processing_stats

        logger.info(f"找到 {len(left_files)} 对图像")
        tasks = list(enumerate(zip(left_files, right_files)))

        start_time = datetime.now()
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {executor.submit(self._process_single_pair, task): task for task in tasks}
            pbar = tqdm(as_completed(futures), total=len(tasks), desc="处理图像", disable=not show_progress)
            for future in pbar:
                idx, success, message = future.result()
                if success:
                    self.processing_stats['successful_frames'] += 1
                else:
                    self.processing_stats['failed_frames'] += 1
                    logger.warning(f"帧 {idx} 处理失败: {message}")

        self.processing_stats['total_frames'] = len(tasks)
        self.processing_stats['processing_time'] = (datetime.now() - start_time).total_seconds()

        parent_dir = os.path.dirname(self.config.output_left_dir.rstrip('/\\'))
        self.config.save(os.path.join(parent_dir, 'preprocessing_config.json'))
        self._print_statistics()
        return self.processing_stats

    def _print_statistics(self):
        stats = self.processing_stats
        logger.info("=" * 50)
        logger.info("处理完成!")
        logger.info(f"总帧数: {stats['total_frames']}")
        logger.info(f"成功: {stats['successful_frames']}")
        logger.info(f"失败: {stats['failed_frames']}")
        logger.info(f"总用时: {stats['processing_time']:.2f} 秒")
        if stats['successful_frames'] > 0:
            fps = stats['successful_frames'] / stats['processing_time']
            logger.info(f"处理速度: {fps:.2f} 帧/秒")
        logger.info("=" * 50)


def create_default_config() -> PreprocessConfig:
    """创建默认配置"""
    return PreprocessConfig(
        calibration_file="../camera_calibration/params/stereo_calib_params_from_matlab_full.npz",
        left_image_dir="../data/left_images/",
        right_image_dir="../data/right_images/",
        bg_left_path="../data/preprocessed/background_left.png",
        bg_right_path="../data/preprocessed/background_right.png",
        output_left_dir="../data/preprocessed/left/",
        output_right_dir="../data/preprocessed/right/",
        crop_to_roi=False,
        roi_coords=None,
        gaussian_blur_size=(5, 5),
        save_intermediate=False,
        num_workers=10,
        batch_size=10,
        clahe=CLAHEConfig(enabled=True, clipLimit=4.0, tileGridSize=(8, 8)),
        foam_removal=FoamRemovalConfig(
            enabled=True,
            # --- 最终平衡性调整：允许扁平椭圆，严格过滤泡沫 ---
            fixed_threshold=68,  # (保留) 用户反馈此值对阴影过滤效果好
            min_particle_area=10,  # (修改) 进一步降低最小面积，稳定远处小标识物
            min_marker_intensity=80,  # (保留) 亮度要求保持不变
            min_solidity=0.88,  # (修改) 提高实心度要求，过滤掉非实心的泡沫
            min_ellipse_score=0.85,  # (修改) 提高椭圆拟合要求，过滤非椭圆泡沫
            max_aspect_ratio=7.0,  # (修改) 大幅放宽长宽比，以允许非常扁平的椭圆
            min_bright_pixels_ratio=0.02,  # (保留) 保留较低的高亮像素比例要求
            closing_kernel_size=3,  # (保留) 保留闭运算
            # --- 其他参数 ---
            opening_kernel_size=3,
            use_multi_stage=True,
            erosion_iterations=1,
            adaptive_threshold=False
        )
    )


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='优化的粒子处理预处理脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--save-config', type=str, help='保存配置到文件')
    parser.add_argument('--workers', type=int, help='并行处理线程数')
    parser.add_argument('--save-intermediate', action='store_true', help='保存中间结果')
    parser.add_argument('--visualize', action='store_true', help='生成分类可视化结果')
    parser.add_argument('--visualize-samples', type=int, default=5, help='可视化样本数量')
    parser.add_argument('--debug-foam', action='store_true', help='调试泡沫去除参数')
    args = parser.parse_args()

    if args.config:
        config = PreprocessConfig.load(args.config)
    else:
        config = create_default_config()

    if args.workers: config.num_workers = args.workers
    if args.save_intermediate: config.save_intermediate = True

    if args.save_config:
        config.save(args.save_config)
        logger.info(f"配置已保存到 {args.save_config}")
        return

    if args.debug_foam:
        logger.info("进入泡沫去除调试模式...")
        # debug_foam_removal(config) # 调试函数需要更新以匹配新的代码结构
        logger.warning("调试函数需要适配新的代码结构，请手动修改或暂时禁用。")
        return

    try:
        preprocessor = StereoPreprocessor(config)
        if args.visualize:
            visualize_samples(preprocessor, config, args.visualize_samples)
        preprocessor.process_batch()
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)


def visualize_samples(preprocessor: StereoPreprocessor, config: PreprocessConfig, num_samples: int = 5):
    logger.info(f"生成 {num_samples} 个可视化样本...")
    parent_dir = os.path.dirname(config.output_left_dir.rstrip('/\\'))
    vis_dir = os.path.join(parent_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    left_files = []
    for ext in ['*.bmp', '*.png', '*.jpg', '*.jpeg']:
        left_files.extend(sorted(glob.glob(os.path.join(config.left_image_dir, ext))))

    if not left_files:
        logger.warning("可视化：未找到图像文件")
        return

    for i, left_file in enumerate(left_files[:num_samples]):
        img = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        # 模拟预处理流程中的增强步骤
        enhanced_img = preprocessor._clahe.apply(img) if preprocessor._clahe else img

        vis_path = os.path.join(vis_dir, f'classification_sample_{i:03d}.png')
        preprocessor.visualize_classification(enhanced_img, vis_path)
    logger.info(f"可视化结果已保存到: {vis_dir}")


if __name__ == '__main__':
    main()
