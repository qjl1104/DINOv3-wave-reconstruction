# particle_processing/02_particle_detection_optimized.py
import cv2
import numpy as np
import glob
import os
import pickle
import sys
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp


@dataclass
class DetectionParams:
    """粒子检测参数数据类"""
    minArea: float = 25
    maxArea: float = 300
    minCircularity: float = 0.4
    minConvexity: float = 0.87
    minInertiaRatio: float = 0.3

    @classmethod
    def from_dict(cls, params_dict: Dict):
        """从字典创建参数实例"""
        return cls(**params_dict)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'minArea': self.minArea,
            'maxArea': self.maxArea,
            'minCircularity': self.minCircularity,
            'minConvexity': self.minConvexity,
            'minInertiaRatio': self.minInertiaRatio
        }


class ParticleDetector:
    """粒子检测器类"""

    def __init__(self, params: DetectionParams):
        """初始化检测器"""
        self.params = params
        self.detector = self._create_detector()

    def _create_detector(self) -> cv2.SimpleBlobDetector:
        """创建并配置Blob检测器"""
        blob_params = cv2.SimpleBlobDetector_Params()

        blob_params.filterByColor = True
        blob_params.blobColor = 255

        blob_params.filterByArea = True
        blob_params.minArea = self.params.minArea
        blob_params.maxArea = self.params.maxArea

        blob_params.filterByCircularity = True
        blob_params.minCircularity = self.params.minCircularity

        blob_params.filterByConvexity = True
        blob_params.minConvexity = self.params.minConvexity

        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = self.params.minInertiaRatio

        return cv2.SimpleBlobDetector_create(blob_params)

    def detect(self, image: np.ndarray) -> Tuple[List[Tuple[float, float]], List]:
        """检测图像中的粒子"""
        if image is None:
            return [], []

        keypoints = self.detector.detect(image)
        particle_centers = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        return particle_centers, keypoints

    def update_params(self, params: DetectionParams):
        """更新检测参数"""
        self.params = params
        self.detector = self._create_detector()


class DetectionVisualizer:
    """检测结果可视化器"""

    @staticmethod
    def visualize(image: np.ndarray, keypoints: List, params: Dict,
                  window_name: str, save_path: Optional[str] = None) -> np.ndarray:
        """可视化检测结果"""
        # 创建彩色图像用于可视化
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # 绘制检测到的粒子
        vis_image = cv2.drawKeypoints(
            vis_image, keypoints, np.array([]),
            (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # 添加参数信息
        y0, dy = 30, 20
        for i, (k, v) in enumerate(params.items()):
            y = y0 + i * dy
            cv2.putText(vis_image, f"{k}: {v}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 添加检测数量信息
        cv2.putText(vis_image, f"Detected: {len(keypoints)} particles",
                    (10, y0 + len(params) * dy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow(window_name, vis_image)

        # 保存图像（如果指定了路径）
        if save_path:
            cv2.imwrite(save_path, vis_image)

        return vis_image


class ParticleDetectionPipeline:
    """粒子检测流水线"""

    def __init__(self, config_file: Optional[str] = None):
        """初始化流水线"""
        self.setup_logging()
        self.config = self.load_config(config_file) if config_file else {}

    def setup_logging(self):
        """设置日志记录"""
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"particle_detection_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"无法加载配置文件 {config_file}: {e}")
            return {}

    def save_config(self, config: Dict, output_file: str):
        """保存配置文件"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4)
        self.logger.info(f"配置已保存至 {output_file}")

    def process_single_image(self, args: Tuple[str, ParticleDetector, bool, Dict]) -> Tuple[str, List, List]:
        """处理单张图像"""
        image_path, detector, visualize, params = args

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return image_path, [], []

        particles, keypoints = detector.detect(image)

        if visualize:
            frame_num = os.path.basename(image_path).split('.')[0]
            DetectionVisualizer.visualize(
                image, keypoints, params,
                f"Detection - Frame {frame_num}",
                None
            )

        return image_path, particles, keypoints

    def process_camera_parallel(self, image_files: List[str], detector: ParticleDetector,
                                camera_name: str, show_live: bool = False,
                                num_workers: int = None) -> List[List[Tuple[float, float]]]:
        """并行处理一个相机的所有图像"""
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)

        self.logger.info(f"使用 {num_workers} 个工作线程处理 {camera_name} 相机图像")

        # 准备参数
        process_args = [
            (img_file, detector, False, detector.params.to_dict())
            for img_file in image_files
        ]

        detections_all_frames = [[] for _ in range(len(image_files))]
        total_detections = 0

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.process_single_image, args): i
                for i, args in enumerate(process_args)
            }

            # 使用进度条显示进度
            with tqdm(total=len(image_files), desc=f"处理 {camera_name} 相机") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        _, particles, keypoints = future.result()
                        detections_all_frames[idx] = particles
                        total_detections += len(particles)

                        # 实时可视化（如果启用）
                        if show_live and len(particles) > 0:
                            image = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE)
                            DetectionVisualizer.visualize(
                                image, keypoints, detector.params.to_dict(),
                                f"{camera_name} Camera Detections"
                            )
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.logger.info("用户中断处理")
                                break

                    except Exception as e:
                        self.logger.error(f"处理图像 {image_files[idx]} 时出错: {e}")

                    pbar.update(1)

        self.logger.info(f"{camera_name} 相机总检测点数: {total_detections}")
        return detections_all_frames

    def compute_statistics(self, detections: List[List[Tuple[float, float]]]) -> Dict:
        """计算检测统计信息"""
        num_frames = len(detections)
        num_detections_per_frame = [len(d) for d in detections]
        total_detections = sum(num_detections_per_frame)

        stats = {
            'total_frames': num_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / num_frames if num_frames > 0 else 0,
            'max_detections_in_frame': max(num_detections_per_frame) if num_detections_per_frame else 0,
            'min_detections_in_frame': min(num_detections_per_frame) if num_detections_per_frame else 0,
            'frames_with_no_detections': sum(1 for n in num_detections_per_frame if n == 0)
        }

        return stats

    def save_results(self, detections: List[List[Tuple[float, float]]],
                     output_file: str, camera_name: str):
        """保存检测结果"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 计算统计信息
        stats = self.compute_statistics(detections)

        # 保存检测结果和统计信息
        results = {
            'detections': detections,
            'statistics': stats,
            'timestamp': datetime.now().isoformat(),
            'camera': camera_name
        }

        with open(output_file, 'wb') as f:
            pickle.dump(results, f)

        # 同时保存统计信息为JSON格式（便于查看）
        stats_file = output_file.replace('.pkl', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)

        self.logger.info(f"{camera_name} 相机检测结果已保存至 {output_file}")
        self.logger.info(f"统计信息: {stats}")

    def run(self, preprocessed_dir_left: str, preprocessed_dir_right: str,
            output_file_left: str, output_file_right: str,
            params_left: Dict, params_right: Dict,
            show_live_video: bool = False, num_workers: int = None):
        """运行完整的检测流程"""

        self.logger.info("=" * 60)
        self.logger.info("开始粒子检测流程")
        self.logger.info("=" * 60)

        # 获取图像文件列表
        left_files = sorted(glob.glob(os.path.join(preprocessed_dir_left, '*.png')))
        right_files = sorted(glob.glob(os.path.join(preprocessed_dir_right, '*.png')))

        # 验证文件
        if not left_files or not right_files:
            self.logger.error("未找到预处理图像文件")
            return

        if len(left_files) != len(right_files):
            self.logger.warning(f"左右相机图像数量不匹配: {len(left_files)} vs {len(right_files)}")

        self.logger.info(f"找到 {len(left_files)} 张左侧图像, {len(right_files)} 张右侧图像")

        # 创建检测器
        detector_left = ParticleDetector(DetectionParams.from_dict(params_left))
        detector_right = ParticleDetector(DetectionParams.from_dict(params_right))

        # 保存使用的参数
        config = {
            'left_camera': params_left,
            'right_camera': params_right,
            'timestamp': datetime.now().isoformat()
        }
        config_file = "../data/detections/detection_params.json"
        self.save_config(config, config_file)

        try:
            # 处理左侧相机
            self.logger.info("\n处理左侧相机...")
            detections_left = self.process_camera_parallel(
                left_files, detector_left, "左侧", show_live_video, num_workers
            )
            self.save_results(detections_left, output_file_left, "左侧")

            # 处理右侧相机
            self.logger.info("\n处理右侧相机...")
            detections_right = self.process_camera_parallel(
                right_files, detector_right, "右侧", show_live_video, num_workers
            )
            self.save_results(detections_right, output_file_right, "右侧")

        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
        except Exception as e:
            self.logger.error(f"处理过程中出错: {e}", exc_info=True)
        finally:
            cv2.destroyAllWindows()

        self.logger.info("\n检测流程完成!")


def main():
    """主函数"""
    # 设置路径
    prep_left_dir = "../data/preprocessed/left/"
    prep_right_dir = "../data/preprocessed/right/"
    out_det_left_file = "../data/detections/detections_left.pkl"
    out_det_right_file = "../data/detections/detections_right.pkl"

    # 检测参数
    detection_params_left = {
        'minArea': 25,
        'maxArea': 300,
        'minCircularity': 0.4,
        'minConvexity': 0.87,
        'minInertiaRatio': 0.3
    }

    detection_params_right = {
        'minArea': 25,
        'maxArea': 300,
        'minCircularity': 0.4,
        'minConvexity': 0.87,
        'minInertiaRatio': 0.3
    }

    # 创建并运行流水线
    pipeline = ParticleDetectionPipeline()
    pipeline.run(
        prep_left_dir, prep_right_dir,
        out_det_left_file, out_det_right_file,
        detection_params_left, detection_params_right,
        show_live_video=True,
        num_workers=4  # 可以根据CPU核心数调整
    )


if __name__ == '__main__':
    main()