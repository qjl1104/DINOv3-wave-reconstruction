# particle_processing/03_trajectory_tracking_2d_KLF_opus_pro.py
import numpy as np
import pickle
import json
import os
import cv2
import glob
import logging
from datetime import datetime
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Union
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d
import warnings

# 忽略运行时警告
warnings.filterwarnings('ignore')


@dataclass
class TrackingConfig:
    """跟踪配置数据类"""
    max_age: int = 20
    min_hits_to_confirm: int = 5
    dist_thresh: float = 80.0
    enable_track_merging: bool = True
    merge_distance_thresh: float = 30.0
    merge_time_gap_thresh: int = 10
    quality_score_thresh: float = 0.3
    high_quality_thresh: float = 0.5
    min_track_length: int = 10

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


class KalmanFilterBase(ABC):
    """卡尔曼滤波器基类"""

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, measurement: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass


class OptimizedExtendedKalmanFilter(KalmanFilterBase):
    """优化的扩展卡尔曼滤波器"""

    def __init__(self, initial_pos: np.ndarray, dt: float = 1.0):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
        self.P[0:2, 0:2] *= 10.0
        self.P[2:4, 2:4] *= 100.0
        self.Q = np.eye(4, dtype=np.float64)
        self.Q[0:2, 0:2] *= 5.0
        self.Q[2:4, 2:4] *= 50.0
        self.R = np.eye(2, dtype=np.float64) * 10.0
        self.dt = dt
        self.velocity_history = deque(maxlen=10)
        self.innovation_history = deque(maxlen=5)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.I = np.eye(4, dtype=np.float64)

    def predict(self) -> np.ndarray:
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 0.95, 0],
            [0, 0, 0, 0.95]
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2]

    def update(self, measurement: np.ndarray) -> None:
        z = measurement.reshape(2, 1)
        y = z - self.H @ self.x.reshape(4, 1)
        self.innovation_history.append(np.linalg.norm(y))
        R_adaptive = self._adaptive_measurement_noise()
        S = self.H @ self.P @ self.H.T + R_adaptive
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (self.I - K @ self.H) @ self.P
        self.velocity_history.append(self.x[2:4].copy())

    def _adaptive_measurement_noise(self) -> np.ndarray:
        if len(self.innovation_history) > 0:
            avg_innovation = np.mean(self.innovation_history)
            if avg_innovation > 50:
                scale = 1 + (avg_innovation - 50) / 50
                return self.R * scale
        return self.R

    def get_state(self) -> np.ndarray:
        return self.x.copy()


class InteractingMultipleModel(KalmanFilterBase):
    """交互式多模型滤波器"""

    def __init__(self, initial_pos: np.ndarray, dt: float = 1.0):
        self.dt = dt
        self.models = {
            'cv': ConstantVelocityFilter(initial_pos, dt),
            'ca': ConstantAccelerationFilter(initial_pos, dt),
            'ct': CoordinatedTurnFilter(initial_pos, dt)
        }
        self.mu = np.array([0.3, 0.3, 0.4])
        self.trans_prob = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.15, 0.15, 0.7]
        ])
        self.x = np.zeros(6)
        self.P = np.eye(6) * 100

    def predict(self) -> np.ndarray:
        c_bar = self.mu @ self.trans_prob
        for i, (model_name, model) in enumerate(self.models.items()):
            w = (self.trans_prob[:, i] * self.mu) / c_bar[i]
            model.predict()
        self.mu = c_bar
        predictions = [model.predict() for model in self.models.values()]
        mixed_pred = sum(self.mu[i] * pred for i, pred in enumerate(predictions))
        return mixed_pred

    def update(self, measurement: np.ndarray) -> None:
        likelihoods = []
        for model in self.models.values():
            model.update(measurement)
            innovation = measurement - model.get_state()[:2]
            likelihood = np.exp(-0.5 * np.dot(innovation, innovation) / 100)
            likelihoods.append(likelihood)
        likelihoods = np.array(likelihoods)
        self.mu = self.mu * likelihoods
        if self.mu.sum() > 0:
            self.mu /= self.mu.sum()

    def get_state(self) -> np.ndarray:
        states = [model.get_state() for model in self.models.values()]
        best_model_idx = np.argmax(self.mu)
        return states[best_model_idx]


class ConstantVelocityFilter(KalmanFilterBase):
    """恒速模型滤波器"""

    def __init__(self, initial_pos: np.ndarray, dt: float = 1.0):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 100
        self.Q = np.eye(4) * 10
        self.R = np.eye(2) * 10
        self.dt = dt
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def predict(self) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2]

    def update(self, measurement: np.ndarray) -> None:
        z = measurement.reshape(2, 1)
        y = z - self.H @ self.x.reshape(4, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x.copy()


class ConstantAccelerationFilter(KalmanFilterBase):
    """恒加速度模型滤波器"""

    def __init__(self, initial_pos: np.ndarray, dt: float = 1.0):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(6) * 100
        self.Q = np.eye(6) * 5
        self.R = np.eye(2) * 10
        self.dt = dt
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    def predict(self) -> np.ndarray:
        F = np.eye(6)
        F[0, 2] = self.dt
        F[0, 4] = 0.5 * self.dt ** 2
        F[1, 3] = self.dt
        F[1, 5] = 0.5 * self.dt ** 2
        F[2, 4] = self.dt
        F[3, 5] = self.dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2]

    def update(self, measurement: np.ndarray) -> None:
        z = measurement.reshape(2, 1)
        y = z - self.H @ self.x.reshape(6, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x.copy()


class CoordinatedTurnFilter(KalmanFilterBase):
    """协调转弯模型滤波器"""

    def __init__(self, initial_pos: np.ndarray, dt: float = 1.0):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(5) * 100
        self.Q = np.eye(5) * 20
        self.R = np.eye(2) * 10
        self.dt = dt
        self.H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])

    def predict(self) -> np.ndarray:
        omega = self.x[4]
        if abs(omega) < 1e-6:
            F = np.eye(5)
            F[0, 2] = self.dt
            F[1, 3] = self.dt
        else:
            s_omega = np.sin(omega * self.dt)
            c_omega = np.cos(omega * self.dt)
            F = np.array([
                [1, 0, s_omega / omega, -(1 - c_omega) / omega, 0],
                [0, 1, (1 - c_omega) / omega, s_omega / omega, 0],
                [0, 0, c_omega, -s_omega, 0],
                [0, 0, s_omega, c_omega, 0],
                [0, 0, 0, 0, 1]
            ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2]

    def update(self, measurement: np.ndarray) -> None:
        z = measurement.reshape(2, 1)
        y = z - self.H @ self.x.reshape(5, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(5) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x.copy()


@dataclass
class Track:
    """增强的轨迹类"""
    id: int
    filter: KalmanFilterBase
    points: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    predictions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    velocities: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    age: int = 0
    consecutive_misses: int = 0
    total_visible_count: int = 1
    last_frame_seen: int = 0
    confidence: float = 1.0
    smoothness_score: float = 1.0
    _features: Optional[np.ndarray] = None

    def predict(self) -> np.ndarray:
        pred_pos = self.filter.predict()
        return pred_pos

    def update(self, detection: Tuple[float, float], frame_idx: int) -> None:
        if hasattr(self, 'last_prediction'):
            pred_error = np.linalg.norm(np.array(detection) - self.last_prediction)
            self.confidence = self._update_confidence(pred_error)
        self.filter.update(np.array(detection))
        self.points[frame_idx] = detection
        state = self.filter.get_state()
        if len(state) >= 4:
            self.velocities[frame_idx] = (state[2], state[3])
        self.age = 0
        self.consecutive_misses = 0
        self.total_visible_count += 1
        self.last_frame_seen = frame_idx
        self._update_smoothness_score()

    def _update_confidence(self, pred_error: float) -> float:
        if pred_error < 20:
            return min(1.0, self.confidence * 1.1)
        elif pred_error < 50:
            return self.confidence
        else:
            return max(0.1, self.confidence * 0.9)

    def _update_smoothness_score(self) -> None:
        if len(self.points) >= 3:
            recent_points = list(self.points.values())[-5:]
            if len(recent_points) >= 3:
                accelerations = []
                for i in range(2, len(recent_points)):
                    p1 = np.array(recent_points[i - 2])
                    p2 = np.array(recent_points[i - 1])
                    p3 = np.array(recent_points[i])
                    v1 = p2 - p1
                    v2 = p3 - p2
                    acc = np.linalg.norm(v2 - v1)
                    accelerations.append(acc)
                if accelerations:
                    avg_acc = np.mean(accelerations)
                    self.smoothness_score = 1.0 / (1.0 + avg_acc / 10.0)

    def mark_missed(self) -> None:
        self.age += 1
        self.consecutive_misses += 1
        self.confidence *= 0.95

    def get_ordered_points(self) -> List[Tuple[float, float]]:
        return [self.points[fi] for fi in sorted(self.points.keys())]

    def get_features(self) -> np.ndarray:
        if self._features is None or len(self.velocities) > len(self._features):
            if len(self.velocities) >= 2:
                velocities = list(self.velocities.values())
                avg_velocity = np.mean(velocities, axis=0)
                velocity_std = np.std(velocities, axis=0)
                self._features = np.concatenate([
                    avg_velocity,
                    velocity_std,
                    [self.confidence, self.smoothness_score]
                ])
            else:
                self._features = np.zeros(6)
        return self._features

    def is_tentative(self, min_hits: int = 3) -> bool:
        return self.total_visible_count < min_hits

    def is_lost(self, max_age: int = 5) -> bool:
        dynamic_max_age = max_age * (1 + self.confidence)
        return self.age > dynamic_max_age or self.consecutive_misses > max_age * 2

    def get_quality_score(self) -> float:
        length_score = min(1.0, len(self.points) / 100.0)
        if self.last_frame_seen > min(self.points.keys()):
            visibility_score = self.total_visible_count / (self.last_frame_seen - min(self.points.keys()) + 1)
        else:
            visibility_score = 1.0
        return self.confidence * self.smoothness_score * length_score * visibility_score

    def interpolate_missing_points(self) -> Dict[int, Tuple[float, float]]:
        if len(self.points) < 2:
            return self.points
        frames = sorted(self.points.keys())
        points_array = np.array([self.points[f] for f in frames])
        interp_x = interp1d(frames, points_array[:, 0], kind='linear', fill_value='extrapolate')
        interp_y = interp1d(frames, points_array[:, 1], kind='linear', fill_value='extrapolate')
        interpolated = {}
        for frame in range(frames[0], frames[-1] + 1):
            if frame in self.points:
                interpolated[frame] = self.points[frame]
            else:
                interpolated[frame] = (float(interp_x(frame)), float(interp_y(frame)))
        return interpolated


class TrackingOptimizer:
    """轨迹跟踪优化器"""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def compute_cost_matrix(self, tracks: List[Track], detections: List[Tuple[float, float]],
                            predicted_positions: List[np.ndarray]) -> np.ndarray:
        """计算高级成本矩阵"""
        num_tracks = len(tracks)
        num_detections = len(detections)
        if num_tracks == 0 or num_detections == 0:
            return np.array([])

        try:
            pred_pos_array = np.array(predicted_positions)
        except ValueError as e:
            self.logger.error(f"无法创建 predicted_positions 数组: {e}")
            return np.array([])

        det_pos_array = np.array(detections)
        dist_matrix = distance_matrix(pred_pos_array, det_pos_array)
        cost_matrix = np.full((num_tracks, num_detections), np.inf)

        for t_idx, track in enumerate(tracks):
            adaptive_thresh = self.config.dist_thresh * (1 + 0.5 * track.confidence)
            valid_indices = np.where(dist_matrix[t_idx] < adaptive_thresh)[0]
            for d_idx in valid_indices:
                dist_cost = dist_matrix[t_idx, d_idx]
                quality_penalty = (1 - track.confidence) * 20
                feature_cost = 0
                if len(track.velocities) > 0:
                    last_vel = list(track.velocities.values())[-1]
                    expected_pos = predicted_positions[t_idx] + np.array(last_vel)
                    feature_cost = np.linalg.norm(expected_pos - detections[d_idx]) * 0.1
                cost_matrix[t_idx, d_idx] = dist_cost + quality_penalty + feature_cost
        return cost_matrix

    def merge_tracks(self, track1: Track, track2: Track) -> Track:
        if track1.get_quality_score() >= track2.get_quality_score():
            main_track, merge_track = track1, track2
        else:
            main_track, merge_track = track2, track1
        main_track.points.update(merge_track.points)
        main_track.velocities.update(merge_track.velocities)
        main_track.total_visible_count += merge_track.total_visible_count
        main_track.last_frame_seen = max(main_track.last_frame_seen, merge_track.last_frame_seen)
        w1 = len(track1.points) / (len(track1.points) + len(track2.points))
        w2 = 1 - w1
        main_track.confidence = w1 * track1.confidence + w2 * track2.confidence
        main_track._update_smoothness_score()
        return main_track

    def should_merge_tracks(self, track1: Track, track2: Track) -> bool:
        frames1 = set(track1.points.keys())
        frames2 = set(track2.points.keys())
        if frames1.intersection(frames2):
            return False
        if frames1 and frames2:
            gap = abs(max(frames1) - min(frames2)) if max(frames1) < min(frames2) else abs(max(frames2) - min(frames1))
            if gap > self.config.merge_time_gap_thresh:
                return False
            if max(frames1) < min(frames2):
                last_point1 = track1.points[max(frames1)]
                first_point2 = track2.points[min(frames2)]
            else:
                last_point1 = track2.points[max(frames2)]
                first_point2 = track1.points[min(frames1)]
            distance = np.linalg.norm(np.array(last_point1) - np.array(first_point2))
            distance_threshold = self.config.merge_distance_thresh * (1 + gap / 5)
            return distance < distance_threshold
        return False


class ParticleTracker:
    """主粒子跟踪器"""

    def __init__(self, config: TrackingConfig, filter_type: str = 'optimized_ekf'):
        self.config = config
        self.filter_type = filter_type
        self.optimizer = TrackingOptimizer(config)
        self.active_tracks: List[Track] = []
        self.completed_tracks: List[Track] = []
        self.next_track_id = 0
        self.stats = defaultdict(int)
        self.frame_idx = 0

    def create_filter(self, initial_pos: Tuple[float, float]) -> KalmanFilterBase:
        pos_array = np.array(initial_pos)
        if self.filter_type == 'optimized_ekf':
            return OptimizedExtendedKalmanFilter(pos_array)
        elif self.filter_type == 'imm':
            return InteractingMultipleModel(pos_array)
        elif self.filter_type == 'cv':
            return ConstantVelocityFilter(pos_array)
        elif self.filter_type == 'ca':
            return ConstantAccelerationFilter(pos_array)
        elif self.filter_type == 'ct':
            return CoordinatedTurnFilter(pos_array)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

    def process_frame(self, frame_idx: int, detections: List[Tuple[float, float]]) -> None:
        self.frame_idx = frame_idx
        predicted_positions = []
        for track in self.active_tracks:
            pred_pos = track.predict()
            track.last_prediction = pred_pos
            predicted_positions.append(pred_pos)
        matched_track_indices, matched_detection_indices = self._data_association(
            detections, predicted_positions
        )
        for track_idx, det_idx in zip(matched_track_indices, matched_detection_indices):
            self.active_tracks[track_idx].update(detections[det_idx], frame_idx)
        unmatched_track_indices = set(range(len(self.active_tracks))) - set(matched_track_indices)
        self._handle_unmatched_tracks(unmatched_indices=unmatched_track_indices)
        unmatched_detection_indices = set(range(len(detections))) - set(matched_detection_indices)
        self._create_new_tracks(detections, unmatched_detection_indices, frame_idx)
        if self.config.enable_track_merging:
            self._merge_tracks()
        self.stats['max_concurrent'] = max(self.stats['max_concurrent'], len(self.active_tracks))

    def _data_association(self, detections: List[Tuple[float, float]],
                          predicted_positions: List[np.ndarray]) -> Tuple[List[int], List[int]]:
        if not self.active_tracks or not detections:
            return [], []

        cost_matrix = self.optimizer.compute_cost_matrix(
            tracks=self.active_tracks,
            detections=detections,
            predicted_positions=predicted_positions
        )

        # 核心修改: 检查成本矩阵的可行性，防止崩溃
        if cost_matrix.size == 0 or not np.any(np.isfinite(cost_matrix)):
            self.logger.warning(f"帧 {self.frame_idx}: 成本矩阵不可行。跳过数据关联。")
            return [], []

        matched_tracks = []
        matched_detections = []
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if np.isfinite(cost_matrix[r, c]):
                matched_tracks.append(r)
                matched_detections.append(c)
        return matched_tracks, matched_detections
        return matched_tracks, matched_detections

    def _handle_unmatched_tracks(self, unmatched_indices: Set[int]) -> None:
        new_active_tracks = []
        for i, track in enumerate(self.active_tracks):
            if i in unmatched_indices:
                track.mark_missed()
                if track.is_lost(self.config.max_age):
                    if (not track.is_tentative(self.config.min_hits_to_confirm) and
                            track.get_quality_score() > self.config.quality_score_thresh):
                        self.completed_tracks.append(track)
                        self.stats['total_completed'] += 1
                else:
                    new_active_tracks.append(track)
            else:
                new_active_tracks.append(track)
        self.active_tracks = new_active_tracks

    def _create_new_tracks(self, detections: List[Tuple[float, float]],
                           unmatched_indices: Set[int], frame_idx: int) -> None:
        for det_idx in unmatched_indices:
            detection = detections[det_idx]
            too_close = False
            for track in self.active_tracks:
                if track.last_frame_seen == frame_idx and frame_idx in track.points:
                    last_pos = track.points[frame_idx]
                    if np.linalg.norm(np.array(detection) - np.array(last_pos)) < 20:
                        too_close = True
                        break
            if not too_close:
                filter_instance = self.create_filter(detection)
                new_track = Track(
                    id=self.next_track_id,
                    filter=filter_instance,
                    points={frame_idx: detection},
                    last_frame_seen=frame_idx
                )
                self.active_tracks.append(new_track)
                self.next_track_id += 1
                self.stats['total_created'] += 1

    def _merge_tracks(self) -> None:
        if len(self.active_tracks) < 2:
            return
        merged_indices = set()
        for i in range(len(self.active_tracks)):
            if i in merged_indices:
                continue
            for j in range(i + 1, len(self.active_tracks)):
                if j in merged_indices:
                    continue
                if self.optimizer.should_merge_tracks(self.active_tracks[i], self.active_tracks[j]):
                    self.active_tracks[i] = self.optimizer.merge_tracks(
                        self.active_tracks[i], self.active_tracks[j]
                    )
                    merged_indices.add(j)
                    self.stats['total_merged'] += 1
        self.active_tracks = [t for i, t in enumerate(self.active_tracks) if i not in merged_indices]

    def finalize(self) -> List[Track]:
        for track in self.active_tracks:
            if (not track.is_tentative(self.config.min_hits_to_confirm) and
                    track.get_quality_score() > self.config.quality_score_thresh):
                self.completed_tracks.append(track)
        high_quality_tracks = [
            t for t in self.completed_tracks
            if len(t.points) >= self.config.min_track_length and
               t.get_quality_score() > self.config.high_quality_thresh
        ]
        return high_quality_tracks


class TrackingPipeline:
    """跟踪流水线"""

    def __init__(self, config: Optional[Union[str, Dict, TrackingConfig]] = None):
        self.setup_logging()
        if isinstance(config, str):
            self.config = self.load_config_from_file(config)
        elif isinstance(config, dict):
            self.config = TrackingConfig.from_dict(config)
        elif isinstance(config, TrackingConfig):
            self.config = config
        else:
            self.config = TrackingConfig()

    def setup_logging(self):
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"trajectory_tracking_{timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config_from_file(self, config_file: str) -> TrackingConfig:
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            return TrackingConfig.from_dict(config_dict.get('tracking_params', {}))
        except Exception as e:
            self.logger.warning(f"无法加载配置文件 {config_file}: {e}")
            return TrackingConfig()

    def track_particles(self, detections: List[List[Tuple[float, float]]],
                        filter_type: str = 'optimized_ekf',
                        show_progress: bool = True) -> List[Track]:
        tracker = ParticleTracker(self.config, filter_type)
        total_frames = len(detections)
        self.logger.info(f"开始跟踪，共 {total_frames} 帧")
        for frame_idx, frame_detections in enumerate(detections):
            tracker.process_frame(frame_idx, frame_detections)
            if show_progress and (frame_idx + 1) % 100 == 0:
                self.logger.info(
                    f"已处理 {frame_idx + 1}/{total_frames} 帧。"
                    f"活跃: {len(tracker.active_tracks)}, "
                    f"完成: {tracker.stats['total_completed']}"
                )
        high_quality_tracks = tracker.finalize()
        self._print_statistics(tracker.stats, high_quality_tracks)
        return high_quality_tracks

    def _print_statistics(self, stats: Dict[str, int], tracks: List[Track]) -> None:
        self.logger.info("\n=== 跟踪统计 ===")
        self.logger.info(f"总创建轨迹数: {stats['total_created']}")
        self.logger.info(f"总完成轨迹数: {stats['total_completed']}")
        self.logger.info(f"高质量轨迹数: {len(tracks)}")
        self.logger.info(f"总合并次数: {stats['total_merged']}")
        self.logger.info(f"最大并发轨迹数: {stats['max_concurrent']}")
        if tracks:
            lengths = [len(t.points) for t in tracks]
            qualities = [t.get_quality_score() for t in tracks]
            self.logger.info(f"\n轨迹长度 - 平均: {np.mean(lengths):.1f}, "
                             f"最小: {min(lengths)}, 最大: {max(lengths)}")
            self.logger.info(f"轨迹质量 - 平均: {np.mean(qualities):.3f}, "
                             f"最小: {min(qualities):.3f}, 最大: {max(qualities):.3f}")

    def save_results(self, tracks: List[Track], output_file: str) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_data = {
            'tracks': tracks,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'num_tracks': len(tracks),
                'avg_length': np.mean([len(t.points) for t in tracks]) if tracks else 0,
                'avg_quality': np.mean([t.get_quality_score() for t in tracks]) if tracks else 0
            }
        }
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        self.logger.info(f"结果已保存至 {output_file}")


def visualize_tracking_results(image_files: List[str], tracks: List[Track],
                               output_video: Optional[str] = None) -> None:
    if not image_files:
        return
    if output_video:
        first_img = cv2.imread(image_files[0])
        height, width = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))
    for frame_idx, img_file in enumerate(image_files):
        img = cv2.imread(img_file)
        if img is None:
            continue
        for track in tracks:
            if frame_idx in track.points:
                pt = track.points[frame_idx]
                pt_int = tuple(map(int, pt))
                quality = track.get_quality_score()
                if quality > 0.7:
                    color = (0, 255, 0)
                elif quality > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                cv2.circle(img, pt_int, 5, color, -1)
                track_frames = sorted([f for f in track.points.keys() if f <= frame_idx])
                if len(track_frames) > 1:
                    for i in range(1, len(track_frames)):
                        if track_frames[i] - track_frames[i - 1] == 1:
                            pt1 = tuple(map(int, track.points[track_frames[i - 1]]))
                            pt2 = tuple(map(int, track.points[track_frames[i]]))
                            cv2.line(img, pt1, pt2, color, 2)
                cv2.putText(img, str(track.id), (pt_int[0] + 5, pt_int[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Tracking Results', img)
        if output_video:
            out.write(img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    if output_video:
        out.release()
        print(f"视频已保存至 {output_video}")


def run_tracking_pipeline(detections_file_left: str, detections_file_right: str,
                          output_file_left: str, output_file_right: str,
                          config: Union[str, Dict, TrackingConfig],
                          filter_type: str = 'optimized_ekf',
                          show_visualization: bool = False,
                          preprocessed_dir_left: Optional[str] = None,
                          preprocessed_dir_right: Optional[str] = None) -> None:
    pipeline = TrackingPipeline(config)
    try:
        with open(detections_file_left, 'rb') as f:
            detections_data_left = pickle.load(f)
        with open(detections_file_right, 'rb') as f:
            detections_data_right = pickle.load(f)

        # 检查并兼容新的数据格式
        if isinstance(detections_data_left, dict) and 'detections' in detections_data_left:
            detections_left = detections_data_left['detections']
        else:
            detections_left = detections_data_left

        if isinstance(detections_data_right, dict) and 'detections' in detections_data_right:
            detections_right = detections_data_right['detections']
        else:
            detections_right = detections_data_right

    except Exception as e:
        pipeline.logger.error(f"加载检测文件失败: {e}")
        return

    pipeline.logger.info("\n=== 处理左侧相机 ===")
    tracks_left = pipeline.track_particles(detections_left, filter_type)
    pipeline.save_results(tracks_left, output_file_left)
    pipeline.logger.info("\n=== 处理右侧相机 ===")
    tracks_right = pipeline.track_particles(detections_right, filter_type)
    pipeline.save_results(tracks_right, output_file_right)

    if show_visualization:
        if preprocessed_dir_left:
            left_files = sorted(glob.glob(os.path.join(preprocessed_dir_left, '*.png')))
            if left_files:
                pipeline.logger.info("显示左侧相机跟踪结果...")
                visualize_tracking_results(left_files, tracks_left)
        if preprocessed_dir_right:
            right_files = sorted(glob.glob(os.path.join(preprocessed_dir_right, '*.png')))
            if right_files:
                pipeline.logger.info("显示右侧相机跟踪结果...")
                visualize_tracking_results(right_files, tracks_right)


def main():
    """主函数"""
    det_left_file = "../data/detections/detections_left.pkl"
    det_right_file = "../data/detections/detections_right.pkl"
    out_traj_left_file = "../data/trajectories/trajectories_2d_left_advanced.pkl"
    out_traj_right_file = "../data/trajectories/trajectories_2d_right_advanced.pkl"
    prep_left_dir = "../data/preprocessed/left/"
    prep_right_dir = "../data/preprocessed/right/"
    config = TrackingConfig(
        max_age=20,
        min_hits_to_confirm=5,
        dist_thresh=80.0,
        enable_track_merging=True,
        merge_distance_thresh=30.0,
        merge_time_gap_thresh=10,
        quality_score_thresh=0.3,
        high_quality_thresh=0.5,
        min_track_length=10
    )
    filter_type = 'imm'
    print(f"使用 {filter_type} 滤波器进行粒子跟踪...")
    run_tracking_pipeline(
        det_left_file, det_right_file,
        out_traj_left_file, out_traj_right_file,
        config=config,
        filter_type=filter_type,
        show_visualization=True,
        preprocessed_dir_left=prep_left_dir,
        preprocessed_dir_right=prep_right_dir
    )


if __name__ == '__main__':
    main()