"""
波浪表面实时三维重建深度学习系统
基于双目视觉的规则波浪自由表面重建
"""

import numpy as np
import torch
torch.backends.cudnn.enabled = False  # 强制关闭 cuDNN
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from scipy.interpolate import RBFInterpolator
from scipy.spatial import distance_matrix
from collections import defaultdict, OrderedDict
import time
from typing import List, Tuple, Dict, Optional
import warnings
from torch.nn import TransformerEncoder, TransformerEncoderLayer
warnings.filterwarnings('ignore')


# ==================== Phase 1: 数据预处理模块 ====================

class CircleDetector:
    """圆片检测器 - 使用组合检测方法"""

    def __init__(self, min_radius=5, max_radius=50, min_area=50, max_area=3000,
                 circularity_threshold=0.4, method='combined'):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold
        self.method = method  # 'contour', 'hough', 'adaptive', 'combined'
        self.debug_mode = False

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        检测图像中的圆片
        返回: [N, 3] array of (x, y, radius)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if self.method == 'combined':
            return self._detect_combined(gray)
        elif self.method == 'contour':
            return self._detect_contour(gray)
        elif self.method == 'adaptive':
            return self._detect_adaptive(gray)
        else:
            return self._detect_hough(gray)

    def _detect_combined(self, gray: np.ndarray) -> np.ndarray:
        """组合检测方法 - 轮廓检测 + 霍夫圆检测"""
        # 使用自适应阈值获得二值图
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 方法1: 轮廓检测
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circles_contour = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > self.circularity_threshold:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if self.min_radius <= radius <= self.max_radius:
                    circles_contour.append([int(x), int(y), int(radius)])

        # 方法2: 霍夫圆检测在二值图上
        circles_hough = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=15,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        # 合并结果
        all_circles = []

        # 添加轮廓检测结果
        if len(circles_contour) > 0:
            all_circles.extend(circles_contour)

        # 添加霍夫检测结果（去重）
        if circles_hough is not None:
            circles_hough = np.round(circles_hough[0, :]).astype("int")
            for ch in circles_hough:
                # 检查是否重复
                is_duplicate = False
                for ac in all_circles:
                    dist = np.sqrt((ch[0] - ac[0]) ** 2 + (ch[1] - ac[1]) ** 2)
                    if dist < 10:  # 距离阈值
                        is_duplicate = True
                        break
                if not is_duplicate:
                    all_circles.append(ch.tolist())

        if self.debug_mode:
            print(f"组合检测: 发现 {len(all_circles)} 个圆片")

        if len(all_circles) > 0:
            return np.array(all_circles)
        return np.array([]).reshape(0, 3)

    def _detect_contour(self, gray: np.ndarray) -> np.ndarray:
        """使用轮廓检测方法"""
        # 使用自适应阈值获得更好的二值化效果
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # 面积筛选
            if area < self.min_area or area > self.max_area:
                continue

            # 计算轮廓的圆度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # 圆度筛选（降低阈值以适应变形的圆片）
            if circularity > self.circularity_threshold:
                # 获取最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)

                # 半径筛选
                if self.min_radius <= radius <= self.max_radius:
                    circles.append([int(x), int(y), int(radius)])

        if self.debug_mode:
            print(f"轮廓检测: 发现 {len(circles)} 个圆片")

        if len(circles) > 0:
            return np.array(circles)
        return np.array([]).reshape(0, 3)

    def _detect_adaptive(self, gray: np.ndarray) -> np.ndarray:
        """使用自适应阈值+霍夫圆检测"""
        # 自适应阈值
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

        # 霍夫圆检测
        circles = cv2.HoughCircles(
            opened,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if self.debug_mode:
                print(f"自适应霍夫检测: 发现 {len(circles)} 个圆片")
            return circles

        return np.array([]).reshape(0, 3)

    def _detect_hough(self, gray: np.ndarray) -> np.ndarray:
        """传统霍夫圆检测（备用）"""
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if self.debug_mode:
                print(f"霍夫检测: 发现 {len(circles)} 个圆片")
            return circles

        return np.array([]).reshape(0, 3)


class StereoMatcher:
    """双目立体匹配器"""

    def __init__(self, camera_params: Dict):
        """
        camera_params: 包含相机内参、外参等标定参数
        """
        self.camera_params = camera_params
        self.baseline = camera_params.get('baseline', 0.12)  # 基线长度 (m)
        self.focal_length = camera_params.get('focal_length', 3000)  # 焦距 (pixels)
        self.cx = camera_params.get('cx', 1280)
        self.cy = camera_params.get('cy', 800)

    def match(self, left_circles: np.ndarray, right_circles: np.ndarray,
              epipolar_threshold: float = 10.0, max_disparity: float = 500.0,
              min_disparity: float = 10.0) -> List[Tuple[int, int]]:
        """
        基于极线约束的圆片匹配

        Args:
            left_circles: 左图圆片 [N, 3] (x, y, radius)
            right_circles: 右图圆片 [M, 3] (x, y, radius)
            epipolar_threshold: 极线约束阈值（像素）
            max_disparity: 最大视差（像素）
            min_disparity: 最小视差（像素）

        返回: 匹配对的索引列表 [(left_idx, right_idx), ...]
        """
        matches = []
        used_right = set()  # 记录已匹配的右图圆片

        if len(left_circles) == 0 or len(right_circles) == 0:
            return matches

        # 为每个左图圆片找最佳匹配
        for i, left_circle in enumerate(left_circles):
            y_left = left_circle[1]
            x_left = left_circle[0]
            r_left = left_circle[2]

            best_match = None
            best_score = float('inf')

            # 在右图中寻找候选匹配
            for j, right_circle in enumerate(right_circles):
                if j in used_right:
                    continue

                y_right = right_circle[1]
                x_right = right_circle[0]
                r_right = right_circle[2]

                # 极线约束：y坐标应该相近
                y_diff = abs(y_left - y_right)
                if y_diff > epipolar_threshold:
                    continue

                # 视差约束：右图x坐标应该小于左图
                disparity = x_left - x_right
                if disparity < min_disparity or disparity > max_disparity:
                    continue

                # 半径相似性
                radius_diff = abs(r_left - r_right) / max(r_left, r_right)
                if radius_diff > 0.5:  # 半径差异不能超过50%
                    continue

                # 计算匹配分数（越小越好）
                score = y_diff + radius_diff * 20  # 权重可调

                if score < best_score:
                    best_score = score
                    best_match = j

            # 添加最佳匹配
            if best_match is not None:
                matches.append((i, best_match))
                used_right.add(best_match)

        return matches

    def triangulate(self, left_circles: np.ndarray, right_circles: np.ndarray,
                    matches: List[Tuple[int, int]]) -> np.ndarray:
        """
        三角测量计算3D坐标
        返回: [N, 3] array of 3D points (x, y, z)
        """
        points_3d = []

        for left_idx, right_idx in matches:
            left_pt = left_circles[left_idx]
            right_pt = right_circles[right_idx]

            # 计算视差
            disparity = left_pt[0] - right_pt[0]

            if disparity > 0:
                # 深度计算: Z = baseline * focal_length / disparity
                z = self.baseline * self.focal_length / disparity

                # 3D坐标
                x = (left_pt[0] - self.cx) * z / self.focal_length
                y = (left_pt[1] - self.cy) * z / self.focal_length

                points_3d.append([x, y, z])

        return np.array(points_3d)


class MultiObjectTracker:
    """多目标跟踪器 - 建立时序一致性"""

    def __init__(self, max_disappeared: int = 3, max_distance: float = 0.05):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance  # 最大匹配距离 (m)

    def update(self, points_3d: np.ndarray) -> Dict[int, np.ndarray]:
        """
        更新跟踪器
        返回: {track_id: [x, y, z]} 的字典
        """
        if len(points_3d) == 0:
            # 标记所有现有对象消失
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return self.objects

        # 如果没有现有对象，注册所有新点
        if len(self.objects) == 0:
            for point in points_3d:
                self._register(point)
        else:
            # 关联现有对象与新检测
            object_ids = list(self.objects.keys())
            object_points = np.array(list(self.objects.values()))

            # 计算距离矩阵
            distances = distance_matrix(object_points[:, :2], points_3d[:, :2])

            # 贪心匹配
            matched_objects = set()
            matched_detections = set()

            for _ in range(min(len(object_ids), len(points_3d))):
                min_idx = np.unravel_index(distances.argmin(), distances.shape)
                obj_idx, det_idx = min_idx

                if distances[obj_idx, det_idx] < self.max_distance:
                    object_id = object_ids[obj_idx]
                    self.objects[object_id] = points_3d[det_idx]
                    self.disappeared[object_id] = 0

                    matched_objects.add(obj_idx)
                    matched_detections.add(det_idx)

                    # 标记为已匹配
                    distances[obj_idx, :] = np.inf
                    distances[:, det_idx] = np.inf
                else:
                    break

            # 处理未匹配的对象
            for obj_idx, object_id in enumerate(object_ids):
                if obj_idx not in matched_objects:
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self._deregister(object_id)

            # 注册新检测
            for det_idx, point in enumerate(points_3d):
                if det_idx not in matched_detections:
                    self._register(point)

        return self.objects.copy()

    def _register(self, point: np.ndarray):
        """注册新对象"""
        self.objects[self.next_object_id] = point
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id: int):
        """注销对象"""
        del self.objects[object_id]
        del self.disappeared[object_id]


def preprocess_stereo_data(left_images: List[np.ndarray],
                           right_images: List[np.ndarray],
                           camera_params: Dict,
                           debug: bool = False) -> List[Dict[int, np.ndarray]]:
    """
    预处理双目图像序列，提取圆片3D轨迹（修正版，支持 debug 参数）
    返回: trajectories_3d: List of dicts per frame: {track_id: np.array([x,y,z])}
    """
    detector = CircleDetector(min_radius=8, max_radius=15)
    stereo_matcher = StereoMatcher(camera_params)
    tracker = MultiObjectTracker(max_disappeared=3)

    # 将 detector 的 debug 模式与外部 debug 参数同步
    detector.debug_mode = debug

    trajectories_3d = []

    num_frames = min(len(left_images), len(right_images))
    if debug:
        print(f"[preprocess_stereo_data] frames to process: {num_frames}")

    for frame_idx in range(num_frames):
        left_img = left_images[frame_idx]
        right_img = right_images[frame_idx]

        # 预处理（可选）：缩放或灰度化已在 detector 内部处理
        try:
            left_circles = detector.detect(left_img)
        except Exception as e:
            if debug:
                print(f"[preprocess_stereo_data] 左图检测异常 (frame {frame_idx}): {e}")
            left_circles = np.array([]).reshape(0, 3)

        try:
            right_circles = detector.detect(right_img)
        except Exception as e:
            if debug:
                print(f"[preprocess_stereo_data] 右图检测异常 (frame {frame_idx}): {e}")
            right_circles = np.array([]).reshape(0, 3)

        if debug:
            print(f"[frame {frame_idx}] left_circles: {len(left_circles)}, right_circles: {len(right_circles)}")

        # 匹配
        matched_pairs = stereo_matcher.match(left_circles, right_circles)

        if debug:
            print(f"[frame {frame_idx}] matched pairs: {len(matched_pairs)}")

        # 三角测量 -> points_3d: Nx3
        points_3d = stereo_matcher.triangulate(left_circles, right_circles, matched_pairs)

        if debug:
            if points_3d is None or len(points_3d) == 0:
                print(f"[frame {frame_idx}] no 3D points triangulated")
            else:
                print(f"[frame {frame_idx}] triangulated {len(points_3d)} points (sample): {points_3d[:3]}")

        # 跟踪器更新（返回 {track_id: np.array([x,y,z])}）
        tracked = tracker.update(points_3d)

        if debug:
            print(f"[frame {frame_idx}] tracked objects: {len(tracked)}")

        trajectories_3d.append(tracked)

    if debug:
        valid = sum(1 for t in trajectories_3d if len(t) > 0)
        print(f"[preprocess_stereo_data] 完成: {len(trajectories_3d)} 帧, 有效帧: {valid}")

    return trajectories_3d


def generate_ground_truth(sparse_points: np.ndarray,
                          grid_size: Tuple[int, int] = (300, 500)) -> np.ndarray:
    """
    基于稀疏3D点生成密集表面Ground Truth
    使用物理约束的RBF插值
    """
    if len(sparse_points) < 3:
        return np.zeros(grid_size)

    # 提取坐标
    x = sparse_points[:, 0]
    y = sparse_points[:, 1]
    z = sparse_points[:, 2]

    # 创建目标网格 (3m×5m, 1cm精度)
    xi = np.linspace(-1.5, 1.5, grid_size[0])  # 3m范围
    yi = np.linspace(-2.5, 2.5, grid_size[1])  # 5m范围
    XI, YI = np.meshgrid(xi, yi, indexing='ij')

    try:
        # RBF插值with smoothing
        rbf = RBFInterpolator(
            np.column_stack([x, y]), z,
            kernel='thin_plate_spline',
            smoothing=0.01  # 物理约束平滑
        )

        ZI = rbf(np.column_stack([XI.ravel(), YI.ravel()])).reshape(grid_size)
    except:
        # 备用方案：简单的最近邻插值
        ZI = np.zeros(grid_size)

    return ZI


# ==================== Phase 2: 网络架构实现 ====================

class ConvLSTMCell(nn.Module):
    """ConvLSTM单元"""

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden_state):
        h, c = hidden_state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Split gates
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """多层ConvLSTM"""

    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, sequence_length):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # 创建多层ConvLSTM
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                kernel_size
            )
            for i in range(num_layers)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        spatial_size = (16, 16)  # 简化的空间维度

        # 初始化隐状态
        h = [torch.zeros(batch_size, self.hidden_dim, *spatial_size).to(x.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim, *spatial_size).to(x.device)
             for _ in range(self.num_layers)]

        # 展开时序
        outputs = []
        for t in range(self.sequence_length):
            x_t = x[:, t].view(batch_size, -1, *spatial_size)

            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer], c[layer] = self.cells[layer](x_t, (h[layer], c[layer]))
                else:
                    h[layer], c[layer] = self.cells[layer](h[layer - 1], (h[layer], c[layer]))

            outputs.append(h[-1])

        return torch.stack(outputs, dim=1)


class PointNet(nn.Module):
    """简化的PointNet用于点云特征提取"""

    def __init__(self, input_dim=3, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x: [batch, num_points, 3]
        batch_size = x.shape[0]
        num_points = x.shape[1]

        x = x.view(batch_size * num_points, -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(batch_size, num_points, -1)

        # 全局最大池化
        x = torch.max(x, dim=1)[0]

        return x


class UNetDecoder(nn.Module):
    """U-Net解码器用于表面重建"""

    def __init__(self, in_channels, out_channels, output_size=(300, 500)):
        super().__init__()
        self.output_size = output_size

        # 编码器
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        # 中间层
        self.middle = self._conv_block(256, 512)

        # 解码器
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)

        # 输出层
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 调整输入大小
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # 中间
        m = self.middle(F.max_pool2d(e3, 2))

        # 解码
        d3 = self.dec3(torch.cat([F.interpolate(m, size=e3.shape[2:], mode='bilinear'), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode='bilinear'), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode='bilinear'), e1], 1))

        # 输出
        out = self.output(d1)
        out = F.interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)

        return out.squeeze(1)


class WaveReconstructionNet(nn.Module):
    """波浪表面重建主网络"""

    def __init__(self, sequence_length=5, num_points=300):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_points = num_points

        # 时序编码器 - 简化版本
        self.temporal_encoder = nn.LSTM(
            input_size=num_points * 3,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.temporal_encoder.flatten_parameters = lambda: None  # 避开 cudnn flatten
        torch.backends.cudnn.enabled = False  # ⚠️ 全局禁用 cuDNN，避免 kernel 报错

        # 点云处理器
        self.point_processor = PointNet(input_dim=3, output_dim=64)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 表面解码器
        self.surface_decoder = UNetDecoder(
            in_channels=1,  # 简化输入
            out_channels=1,
            output_size=(300, 500)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # 时序特征提取
        x_temporal = x.view(batch_size, self.sequence_length, -1)
        lstm_out, _ = self.temporal_encoder(x_temporal)
        temporal_features = lstm_out[:, -1, :]  # 取最后时刻

        # 点云特征提取
        x_points = x[:, -1]  # 最后一帧的点云
        point_features = self.point_processor(x_points)

        # 特征融合
        combined_features = torch.cat([temporal_features, point_features], dim=1)
        fused = self.fusion(combined_features)

        # 重塑为2D特征图
        feature_map = fused.view(batch_size, 1, 16, 16)

        # 表面重建
        surface = self.surface_decoder(feature_map)

        return surface


# ==================== 训练数据集类 ====================

class WaveDataset(Dataset):
    """波浪数据集"""

    def __init__(self, trajectories_3d: List[Dict], sequence_length: int = 5):
        self.sequence_length = sequence_length
        self.samples = []

        # 构建训练样本
        for i in range(sequence_length, len(trajectories_3d)):
            # 准备输入序列
            input_sequence = []
            for j in range(i - sequence_length, i):
                # 将字典转换为固定大小的数组
                points = self._dict_to_array(trajectories_3d[j])
                input_sequence.append(points)

            # 生成目标表面
            current_points = self._dict_to_array(trajectories_3d[i])
            if len(current_points) > 3:
                target_surface = generate_ground_truth(current_points)

                self.samples.append({
                    'input_points': np.array(input_sequence),
                    'target_surface': target_surface
                })

    def _dict_to_array(self, point_dict: Dict, max_points: int = 300) -> np.ndarray:
        """将点字典转换为固定大小的数组"""
        if not point_dict:  # 处理空字典
            return np.zeros((max_points, 3))

        points = np.array(list(point_dict.values()))

        if len(points) == 0:  # 处理空数组
            return np.zeros((max_points, 3))

        # 确保points是2D数组
        if len(points.shape) == 1:
            points = points.reshape(-1, 3)

        # 填充或截断到固定大小
        if len(points) < max_points:
            # 填充
            padded = np.zeros((max_points, 3))
            padded[:len(points)] = points
            return padded
        else:
            # 截断
            return points[:max_points]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'points': torch.FloatTensor(sample['input_points']),
            'surface': torch.FloatTensor(sample['target_surface'])
        }


# ==================== 损失函数 ====================

class GradientLoss(nn.Module):
    """梯度平滑损失"""

    def forward(self, pred):
        dy = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
        dx = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        return torch.mean(dy) + torch.mean(dx)


class WaveReconstructionLoss(nn.Module):
    """波浪重建综合损失"""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.grad_loss = GradientLoss()

    def forward(self, pred_surface, target_surface):
        # 主要重建损失
        recon_loss = self.l1_loss(pred_surface, target_surface)

        # 梯度平滑损失
        smooth_loss = self.grad_loss(pred_surface)

        return recon_loss + 0.1 * smooth_loss


# ==================== Phase 3: 推理优化 ====================

class OptimizedInference:
    """优化的推理引擎"""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = WaveReconstructionNet()
        if model_path and torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        # 预分配缓冲区
        self.input_buffer = torch.zeros(1, 5, 300, 3).to(self.device)
        self.output_buffer = None

        # 推理计时
        self.inference_times = []

    def predict(self, point_sequence: np.ndarray) -> np.ndarray:
        """
        实时推理接口
        输入: [5, 300, 3] 最近5帧的圆片3D坐标
        输出: [300, 500] 密集表面高度图
        """
        with torch.no_grad():
            # 准备输入
            self.input_buffer[0] = torch.from_numpy(point_sequence).float()

            # 计时开始
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            # 推理
            surface = self.model(self.input_buffer)

            # 计时结束
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)

            return surface.cpu().numpy()[0]

    def get_avg_inference_time(self) -> float:
        """获取平均推理时间(ms)"""
        if self.inference_times:
            return np.mean(self.inference_times)
        return 0.0


# ==================== Phase 4: 系统集成 ====================

class WaveReconstructionSystem:
    """完整的波浪重建系统"""

    def __init__(self, camera_params: Dict):
        self.camera_params = camera_params

        # 初始化各模块 - 使用组合检测方法
        self.detector = CircleDetector(
            min_radius=5,
            max_radius=50,
            min_area=50,
            max_area=3000,
            circularity_threshold=0.3,
            method='combined'  # 使用组合检测方法
        )
        self.stereo_matcher = StereoMatcher(camera_params)
        self.tracker = MultiObjectTracker()
        self.inference_engine = None

        # 数据缓冲
        self.point_buffer = []
        self.buffer_size = 5

    def train(self, left_images: List[np.ndarray],
              right_images: List[np.ndarray],
              epochs: int = 50,
              batch_size: int = 8,
              learning_rate: float = 1e-4):
        """训练模型"""
        print("Phase 1: 数据预处理...")
        # 启用调试模式查看更多信息
        trajectories = preprocess_stereo_data(
            left_images, right_images, self.camera_params, debug=True
        )

        print(f"提取到 {len(trajectories)} 帧轨迹数据")

        # 检查是否有足够的有效数据
        valid_frames = sum(1 for t in trajectories if len(t) > 0)
        print(f"包含有效3D点的帧数: {valid_frames}/{len(trajectories)}")

        if valid_frames < 10:
            print("\n错误: 有效数据太少，无法进行训练")
            print("请检查:")
            print("1. 图像中是否有清晰可见的圆片标记")
            print("2. 圆片检测参数是否合适")
            print("3. 相机参数是否正确")
            return None

        # 创建数据集
        dataset = WaveDataset(trajectories, sequence_length=5)

        if len(dataset) == 0:
            print("错误: 无法创建训练数据集")
            return None

        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

        # 初始化模型
        print("\nPhase 2: 网络训练...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        model = WaveReconstructionNet().to(device)

        # 损失函数和优化器
        criterion = WaveReconstructionLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                points = batch['points'].to(device)
                surface = batch['surface'].to(device)

                optimizer.zero_grad()
                pred = model(points)
                loss = criterion(pred, surface)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    points = batch['points'].to(device)
                    surface = batch['surface'].to(device)

                    pred = model(points)
                    loss = criterion(pred, surface)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_wave_model.pth')

            scheduler.step()

        print("\nPhase 3: 推理优化...")
        self.inference_engine = OptimizedInference('best_wave_model.pth')

        return model

    def process_frame(self, left_image: np.ndarray, right_image: np.ndarray) -> Optional[np.ndarray]:
        """
        处理单帧图像对
        返回: 密集表面高度图 [300, 500] 或 None
        """
        # 检测圆片
        left_circles = self.detector.detect(left_image)
        right_circles = self.detector.detect(right_image)

        # 立体匹配
        matches = self.stereo_matcher.match(left_circles, right_circles)

        # 三角测量
        points_3d = self.stereo_matcher.triangulate(left_circles, right_circles, matches)

        # 跟踪
        tracked_points = self.tracker.update(points_3d)

        # 转换为数组
        points_array = self._dict_to_array(tracked_points)

        # 更新缓冲区
        self.point_buffer.append(points_array)
        if len(self.point_buffer) > self.buffer_size:
            self.point_buffer.pop(0)

        # 如果缓冲区满，进行推理
        if len(self.point_buffer) == self.buffer_size and self.inference_engine:
            sequence = np.array(self.point_buffer)
            surface = self.inference_engine.predict(sequence)
            return surface

        return None

    def _dict_to_array(self, point_dict: Dict, max_points: int = 300) -> np.ndarray:
        """将点字典转换为固定大小的数组"""
        points = np.array(list(point_dict.values()))

        if len(points) == 0:
            return np.zeros((max_points, 3))

        if len(points) < max_points:
            padded = np.zeros((max_points, 3))
            padded[:len(points)] = points
            return padded
        else:
            return points[:max_points]

    def evaluate(self, test_images_left: List[np.ndarray],
                 test_images_right: List[np.ndarray]) -> Dict[str, float]:
        """评估系统性能"""
        if not self.inference_engine:
            print("请先训练模型")
            return {}

        metrics = {
            'mae': [],
            'rmse': [],
            'inference_time': []
        }

        # 重置缓冲区
        self.point_buffer = []

        for i, (left_img, right_img) in enumerate(zip(test_images_left, test_images_right)):
            surface = self.process_frame(left_img, right_img)

            if surface is not None:
                # 这里应该与ground truth比较，简化处理
                metrics['mae'].append(np.mean(np.abs(surface)))
                metrics['rmse'].append(np.sqrt(np.mean(surface ** 2)))

        # 获取推理时间
        avg_time = self.inference_engine.get_avg_inference_time()

        return {
            'avg_mae_cm': np.mean(metrics['mae']) * 100 if metrics['mae'] else 0,
            'avg_rmse_cm': np.mean(metrics['rmse']) * 100 if metrics['rmse'] else 0,
            'avg_inference_ms': avg_time,
            'fps': 1000 / avg_time if avg_time > 0 else 0
        }


# ==================== 真实数据加载 ====================

import os
from pathlib import Path


def load_stereo_images(left_dir: str, right_dir: str,
                       max_frames: Optional[int] = None,
                       image_format: str = '*.png') -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    从指定目录加载双目图像数据

    Args:
        left_dir: 左相机图像目录
        right_dir: 右相机图像目录
        max_frames: 最大加载帧数（None表示加载所有）
        image_format: 图像格式，支持 '*.png', '*.jpg', '*.bmp' 等

    Returns:
        (left_images, right_images): 双目图像列表
    """
    left_path = Path(left_dir)
    right_path = Path(right_dir)

    # 获取图像文件列表
    left_files = sorted(left_path.glob(image_format))
    right_files = sorted(right_path.glob(image_format))

    # 确保文件数量匹配
    if len(left_files) != len(right_files):
        print(f"警告: 左右相机图像数量不匹配 ({len(left_files)} vs {len(right_files)})")
        min_count = min(len(left_files), len(right_files))
        left_files = left_files[:min_count]
        right_files = right_files[:min_count]

    # 限制加载数量
    if max_frames is not None:
        left_files = left_files[:max_frames]
        right_files = right_files[:max_frames]

    print(f"找到 {len(left_files)} 对双目图像")

    left_images = []
    right_images = []

    for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
        # 读取图像
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))

        if left_img is None or right_img is None:
            print(f"警告: 无法读取图像对 {i}: {left_file.name}, {right_file.name}")
            continue

        left_images.append(left_img)
        right_images.append(right_img)

        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"已加载 {i + 1}/{len(left_files)} 帧...")

    print(f"成功加载 {len(left_images)} 对图像")
    print(f"图像尺寸: {left_images[0].shape if left_images else 'N/A'}")

    return left_images, right_images


def load_camera_params(npz_path: Optional[str] = None) -> Dict:
    """
    加载相机参数

    Args:
        npz_path: .npz参数文件路径（可选）

    Returns:
        camera_params: 相机参数字典
    """
    # 使用您提供的真实相机参数
    camera_params = {
        'baseline': 1.413219,  # 相机基线 (米)
        'focal_length': 3937.2091,  # 焦距 (像素)
        'cx': 1349.8557,  # 主点x坐标 (像素)
        'cy': 952.7315  # 主点y坐标 (像素)
    }

    # 如果提供了npz文件路径，尝试从文件加载
    if npz_path and os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            print(f"从 {npz_path} 加载相机参数")
            print("可用的参数键:", list(data.keys()))

            # 根据实际的npz文件结构更新参数
            # 这里需要根据您的实际文件结构调整
            if 'baseline' in data:
                camera_params['baseline'] = float(data['baseline'])
            if 'focal_length' in data:
                camera_params['focal_length'] = float(data['focal_length'])
            # ... 其他参数

        except Exception as e:
            print(f"警告: 无法从npz文件加载参数: {e}")
            print("使用默认参数")

    print("\n相机参数:")
    for key, value in camera_params.items():
        print(f"  {key}: {value}")

    return camera_params


def preprocess_real_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    预处理真实图像

    Args:
        images: 原始图像列表

    Returns:
        processed_images: 预处理后的图像
    """
    processed = []

    for img in images:
        # 如果需要，进行预处理
        # 例如：去畸变、直方图均衡化、降噪等

        # 直方图均衡化增强对比度（可选）
        if len(img.shape) == 3:
            # 转换到LAB色彩空间
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # 应用CLAHE到L通道
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # 合并通道
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        processed.append(img)

    return processed


def main():
    """主函数 - 使用真实数据训练和评估"""
    print("=" * 50)
    print("波浪表面实时三维重建系统 - 真实数据版")
    print("=" * 50)

    # 1. 加载相机参数
    print("\n1. 加载相机参数...")
    camera_params = {
        'baseline': 1.413219,  # 相机基线 (米)
        'focal_length': 3937.2091,  # 焦距 (像素)
        'cx': 1349.8557,  # 主点x坐标 (像素)
        'cy': 952.7315  # 主点y坐标 (像素)
    }

    print("\n相机参数:")
    for key, value in camera_params.items():
        print(f"  {key}: {value}")

    # 2. 加载真实图像数据
    print("\n2. 加载双目图像数据...")
    left_dir = r"D:\zuchuan\lresult"  # 左相机图像目录
    right_dir = r"D:\zuchuan\rresult"  # 右相机图像目录

    # 检查目录是否存在
    if not os.path.exists(left_dir):
        print(f"错误: 左相机目录不存在: {left_dir}")
        return

    if not os.path.exists(right_dir):
        print(f"错误: 右相机目录不存在: {right_dir}")
        return

    # 加载图像，文件格式是.bmp
    print("\n3. 加载图像数据（BMP格式）...")
    left_images, right_images = load_stereo_images(
        left_dir, right_dir, max_frames=1000, image_format='*.bmp'
    )

    if len(left_images) == 0:
        print("错误: 未能加载任何BMP图像")
        print("尝试PNG格式...")
        left_images, right_images = load_stereo_images(
            left_dir, right_dir, max_frames=1000, image_format='*.png'
        )

        if len(left_images) == 0:
            print("错误: 未找到任何图像文件")
            return

    print(f"成功加载 {len(left_images)} 对图像")

    # 4. 创建系统实例
    print("\n4. 初始化波浪重建系统...")
    print("使用组合检测方法（轮廓+霍夫圆，基于测试可检测500+圆片）")
    system = WaveReconstructionSystem(camera_params)

    # 5. 数据划分
    total_frames = len(left_images)
    train_size = int(0.8 * total_frames)
    val_size = int(0.1 * total_frames)
    test_size = total_frames - train_size - val_size

    print(f"\n5. 数据划分:")
    print(f"  总帧数: {total_frames}")
    print(f"  训练集: {train_size} 帧")
    print(f"  验证集: {val_size} 帧")
    print(f"  测试集: {test_size} 帧")

    train_left = left_images[:train_size]
    train_right = right_images[:train_size]

    test_left = left_images[train_size + val_size:]
    test_right = right_images[train_size + val_size:]

    # 6. 训练模型
    print("\n6. 开始训练模型...")
    print("预期每帧检测约500个圆片（基于测试结果）")
    print("这可能需要较长时间，请耐心等待...")

    try:
        model = system.train(
            train_left,
            train_right,
            epochs=50,  # 可以根据需要调整
            batch_size=4,  # 减小batch size避免内存问题
            learning_rate=1e-4
        )

        if model is None:
            print("\n训练失败，请检查数据和参数")
            return

        # 7. 评估性能
        print("\n7. 评估系统性能...")
        metrics = system.evaluate(test_left, test_right)

        print("\n" + "=" * 50)
        print("性能指标:")
        print(f"平均绝对误差: {metrics.get('avg_mae_cm', 0):.2f} cm")
        print(f"均方根误差: {metrics.get('avg_rmse_cm', 0):.2f} cm")
        print(f"平均推理时间: {metrics.get('avg_inference_ms', 0):.2f} ms")
        print(f"推理帧率: {metrics.get('fps', 0):.1f} fps")

        # 检查是否达到目标
        if metrics.get('avg_mae_cm', float('inf')) <= 1.0:
            print("✓ 达到1cm精度目标!")
        else:
            print("✗ 未达到1cm精度目标，可能需要更多训练数据或参数调优")

        if metrics.get('fps', 0) >= 30:
            print("✓ 达到30fps实时性目标!")
        else:
            print("✗ 未达到30fps实时性目标，可能需要模型优化或硬件加速")
        print("=" * 50)

        # 8. 保存模型
        print("\n8. 保存训练好的模型...")
        torch.save(model.state_dict(), 'wave_reconstruction_final.pth')
        print("模型已保存至 wave_reconstruction_final.pth")

        # 9. 实时推理演示
        print("\n9. 实时推理演示...")
        demo_count = min(5, len(test_left))
        for i in range(demo_count):
            surface = system.process_frame(test_left[i], test_right[i])
            if surface is not None:
                print(f"测试帧 {i}: 生成 {surface.shape} 表面")
                print(f"  高度范围: [{surface.min():.3f}, {surface.max():.3f}] m")
                print(f"  推理时间: {system.inference_engine.get_avg_inference_time():.2f} ms")

    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

        print("\n调试建议:")
        print("1. 确认圆片检测效果（应该每帧检测到400-500个圆片）")
        print("2. 如果内存不足，尝试减小batch_size到2或1")
        print("3. 如果检测效果不好，可以调整CircleDetector参数")
        print("4. 检查GPU是否正常工作（使用nvidia-smi命令）")

    print("\n系统运行完成！")


def run_inference_only(model_path: str, left_dir: str, right_dir: str):
    """
    仅运行推理（使用已训练好的模型）

    Args:
        model_path: 训练好的模型路径
        left_dir: 左相机图像目录
        right_dir: 右相机图像目录
    """
    print("=" * 50)
    print("波浪表面实时推理")
    print("=" * 50)

    # 加载相机参数
    camera_params = {
        'baseline': 1.413219,
        'focal_length': 3937.2091,
        'cx': 1349.8557,
        'cy': 952.7315
    }

    # 创建系统
    system = WaveReconstructionSystem(camera_params)

    # 加载模型
    system.inference_engine = OptimizedInference(model_path)

    # 加载测试图像
    left_images, right_images = load_stereo_images(
        left_dir, right_dir, max_frames=100
    )

    # 实时处理
    for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
        surface = system.process_frame(left_img, right_img)
        if surface is not None:
            print(f"帧 {i}: 重建完成, 推理时间: "
                  f"{system.inference_engine.get_avg_inference_time():.2f} ms")


if __name__ == "__main__":
    main()

    # 如果只想运行推理（已有训练好的模型）
    # run_inference_only('wave_reconstruction_final.pth',
    #                   r"D:\zuchuan\lresult",
    #                   r"D:\zuchuan\lresult")