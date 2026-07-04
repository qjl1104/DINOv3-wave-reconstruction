#!/usr/bin/env python3
"""
优化的粒子处理预处理脚本
支持多线程处理、更好的错误处理、配置管理和进度显示
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
    """优化的泡沫去除器"""

    def __init__(self, config: FoamRemovalConfig):
        self.config = config
        self._kernel_cache = {}

    def _get_kernel(self, size: int) -> np.ndarray:
        """缓存形态学核"""
        if size not in self._kernel_cache:
            self._kernel_cache[size] = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (size, size)
            )
        return self._kernel_cache[size]

    def process(self, image: np.ndarray) -> np.ndarray:
        """处理图像，去除泡沫和噪点"""
        if not self.config.enabled:
            return image

        # 选择阈值方法
        if self.config.adaptive_threshold:
            # 自适应阈值
            thresh = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config.adaptive_block_size,
                self.config.adaptive_C
            )
        else:
            # 固定阈值
            _, thresh = cv2.threshold(
                image,
                self.config.fixed_threshold,
                255,
                cv2.THRESH_BINARY
            )

        # 形态学开运算
        kernel = self._get_kernel(self.config.opening_kernel_size)
        opened_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            opened_thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 使用numpy向量化操作优化轮廓处理
        output_image = image.copy()

        for contour in contours:
            area = cv2.contourArea(contour)

            # 去除过小的噪点
            if area < self.config.min_particle_area:
                cv2.drawContours(output_image, [contour], -1, 0, thickness=cv2.FILLED)
                continue

            # 去除大块泡沫
            if area > self.config.min_foam_area:
                # 创建掩码
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

                # 计算平均强度
                mean_val = cv2.mean(image, mask=mask)[0]

                # 计算圆度
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    # 根据圆度和强度判断是否为泡沫
                    if (circularity < self.config.max_foam_circularity and
                            mean_val < self.config.max_foam_intensity):
                        cv2.drawContours(output_image, [contour], -1, 0, thickness=cv2.FILLED)

        return output_image


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

        # 统计信息
        self.processing_stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'processing_time': 0.0
        }

    def _setup_clahe(self):
        """设置CLAHE对象"""
        if self.config.clahe.enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe.clipLimit,
                tileGridSize=self.config.clahe.tileGridSize
            )

    def _load_calibration_data(self):
        """加载相机校准数据"""
        try:
            logger.info(f"加载校准数据: {self.config.calibration_file}")
            calib_data = np.load(self.config.calibration_file)
            self.map1_l = calib_data['map1_left']
            self.map2_l = calib_data['map2_left']
            self.map1_r = calib_data['map1_right']
            self.map2_r = calib_data['map2_right']
            logger.info("校准数据加载成功")
        except Exception as e:
            logger.error(f"加载校准数据失败: {e}")
            raise

    def _load_background_models(self):
        """加载背景模型"""
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
        """创建输出目录"""
        Path(self.config.output_left_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_right_dir).mkdir(parents=True, exist_ok=True)

        if self.config.save_intermediate:
            # 创建中间结果目录
            for stage in ['rectified', 'subtracted', 'enhanced', 'foam_removed']:
                Path(self.config.output_left_dir, stage).mkdir(exist_ok=True)
                Path(self.config.output_right_dir, stage).mkdir(exist_ok=True)

    def preprocess_image_pair(
            self,
            img_l_raw: np.ndarray,
            img_r_raw: np.ndarray,
            frame_id: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单对图像

        Args:
            img_l_raw: 左相机原始图像
            img_r_raw: 右相机原始图像
            frame_id: 帧ID（用于保存中间结果）

        Returns:
            处理后的左右图像对
        """
        # 1. 立体校正
        rectified_left = cv2.remap(img_l_raw, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_r_raw, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        # 对背景进行校正
        bg_left_rect = cv2.remap(self.bg_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        bg_right_rect = cv2.remap(self.bg_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        if self.config.save_intermediate and frame_id is not None:
            self._save_intermediate('rectified', frame_id, rectified_left, rectified_right)

        # 2. 转换为灰度图
        if len(rectified_left.shape) == 3:
            gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = rectified_left
            gray_right = rectified_right

        # 3. 背景减除（优化：使用absdiff）
        subtracted_left = cv2.absdiff(gray_left, bg_left_rect)
        subtracted_right = cv2.absdiff(gray_right, bg_right_rect)

        if self.config.save_intermediate and frame_id is not None:
            self._save_intermediate('subtracted', frame_id, subtracted_left, subtracted_right)

        # 4. CLAHE增强
        if self._clahe:
            enhanced_left = self._clahe.apply(subtracted_left)
            enhanced_right = self._clahe.apply(subtracted_right)
        else:
            enhanced_left = subtracted_left
            enhanced_right = subtracted_right

        if self.config.save_intermediate and frame_id is not None:
            self._save_intermediate('enhanced', frame_id, enhanced_left, enhanced_right)

        # 5. 去除泡沫
        foam_removed_left = self.foam_remover.process(enhanced_left)
        foam_removed_right = self.foam_remover.process(enhanced_right)

        if self.config.save_intermediate and frame_id is not None:
            self._save_intermediate('foam_removed', frame_id, foam_removed_left, foam_removed_right)

        # 6. 高斯模糊
        blurred_left = cv2.GaussianBlur(
            foam_removed_left,
            self.config.gaussian_blur_size,
            0
        )
        blurred_right = cv2.GaussianBlur(
            foam_removed_right,
            self.config.gaussian_blur_size,
            0
        )

        # 7. ROI裁剪
        if self.config.crop_to_roi and self.config.roi_coords:
            x, y, w, h = self.config.roi_coords
            blurred_left = blurred_left[y:y + h, x:x + w]
            blurred_right = blurred_right[y:y + h, x:x + w]

        return blurred_left, blurred_right

    def _save_intermediate(self, stage: str, frame_id: int, img_left: np.ndarray, img_right: np.ndarray):
        """保存中间结果"""
        left_path = Path(self.config.output_left_dir, stage, f"frame_{frame_id:05d}.png")
        right_path = Path(self.config.output_right_dir, stage, f"frame_{frame_id:05d}.png")
        cv2.imwrite(str(left_path), img_left)
        cv2.imwrite(str(right_path), img_right)

    def _process_single_pair(self, args: Tuple[int, str, str]) -> Tuple[int, bool, str]:
        """处理单对图像（用于多线程）"""
        idx, left_path, right_path = args

        try:
            # 读取图像
            img_l = cv2.imread(left_path)
            img_r = cv2.imread(right_path)

            if img_l is None or img_r is None:
                return idx, False, f"无法读取图像: {left_path} 或 {right_path}"

            # 处理图像对
            processed_l, processed_r = self.preprocess_image_pair(img_l, img_r, idx)

            # 保存结果
            out_left = Path(self.config.output_left_dir, f"preprocessed_frame_{idx:05d}.png")
            out_right = Path(self.config.output_right_dir, f"preprocessed_frame_{idx:05d}.png")

            cv2.imwrite(str(out_left), processed_l)
            cv2.imwrite(str(out_right), processed_r)

            return idx, True, "成功"

        except Exception as e:
            return idx, False, str(e)

    def process_batch(self, show_progress: bool = True) -> Dict:
        """
        批量处理所有图像对

        Args:
            show_progress: 是否显示进度条

        Returns:
            处理统计信息
        """
        # 获取图像文件列表
        left_pattern = os.path.join(self.config.left_image_dir, '*.bmp')
        right_pattern = os.path.join(self.config.right_image_dir, '*.bmp')

        left_files = sorted(glob.glob(left_pattern))
        right_files = sorted(glob.glob(right_pattern))

        # 支持更多格式
        if not left_files:
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff']:
                left_files.extend(sorted(glob.glob(
                    os.path.join(self.config.left_image_dir, ext)
                )))
                right_files.extend(sorted(glob.glob(
                    os.path.join(self.config.right_image_dir, ext)
                )))

        if not left_files or not right_files:
            logger.error("未找到图像文件")
            return self.processing_stats

        if len(left_files) != len(right_files):
            logger.warning(f"左右图像数量不匹配: {len(left_files)} vs {len(right_files)}")
            min_len = min(len(left_files), len(right_files))
            left_files = left_files[:min_len]
            right_files = right_files[:min_len]

        logger.info(f"找到 {len(left_files)} 对图像")

        # 准备任务列表
        tasks = [(i, l, r) for i, (l, r) in enumerate(zip(left_files, right_files))]

        # 记录开始时间
        import time
        start_time = time.time()

        # 多线程处理
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # 提交任务
            futures = {
                executor.submit(self._process_single_pair, task): task
                for task in tasks
            }

            # 处理结果
            if show_progress:
                pbar = tqdm(total=len(tasks), desc="处理图像")

            for future in as_completed(futures):
                idx, success, message = future.result()

                if success:
                    self.processing_stats['successful_frames'] += 1
                else:
                    self.processing_stats['failed_frames'] += 1
                    logger.warning(f"帧 {idx} 处理失败: {message}")

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        # 更新统计信息
        self.processing_stats['total_frames'] = len(tasks)
        self.processing_stats['processing_time'] = time.time() - start_time

        # 保存处理参数
        config_save_path = Path(self.config.output_left_dir).parent / 'preprocessing_config.json'
        self.config.save(str(config_save_path))
        logger.info(f"配置已保存到: {config_save_path}")

        # 打印统计信息
        self._print_statistics()

        return self.processing_stats

    def _print_statistics(self):
        """打印处理统计信息"""
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
        num_workers=4,
        batch_size=10,
        clahe=CLAHEConfig(
            enabled=True,
            clipLimit=4.0,
            tileGridSize=(8, 8)
        ),
        foam_removal=FoamRemovalConfig(
            enabled=True,
            fixed_threshold=40,
            min_particle_area=15,
            min_foam_area=500,
            max_foam_circularity=0.5,
            max_foam_intensity=150,
            opening_kernel_size=3,
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
    parser.add_argument('--adaptive-threshold', action='store_true', help='使用自适应阈值')
    parser.add_argument('--no-progress', action='store_true', help='不显示进度条')

    args = parser.parse_args()

    # 加载或创建配置
    if args.config:
        config = PreprocessConfig.load(args.config)
        logger.info(f"从 {args.config} 加载配置")
    else:
        config = create_default_config()
        logger.info("使用默认配置")

    # 覆盖命令行参数
    if args.workers:
        config.num_workers = args.workers
    if args.save_intermediate:
        config.save_intermediate = True
    if args.adaptive_threshold:
        config.foam_removal.adaptive_threshold = True

    # 保存配置（如果需要）
    if args.save_config:
        config.save(args.save_config)
        logger.info(f"配置已保存到 {args.save_config}")
        return

    # 创建预处理器并运行
    try:
        preprocessor = StereoPreprocessor(config)
        stats = preprocessor.process_batch(show_progress=not args.no_progress)

        # 保存处理报告
        report_path = Path(
            config.output_left_dir).parent / f'processing_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"处理报告已保存到: {report_path}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise


if __name__ == '__main__':
    main()