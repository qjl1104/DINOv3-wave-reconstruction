# particle_processing/04_trajectory_matching_optimized.py
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from scipy.signal import correlate
import os
from collections import defaultdict
import cv2


# ----------------- 兼容性类定义 -----------------
# 这些类确保能够加载之前保存的轨迹文件

class SimpleKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 1e-1
        self.Q = np.eye(4) * 1e-3
        self.R = np.eye(2) * 1e-1
        self.dt = 1.0


class ImprovedKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0, 0.0], dtype=float)


class ExtendedKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.1, 0.0], dtype=float)


class OptimizedExtendedKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)


class UltraOptimizedKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)


class WaveParticleKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)


class StrictWaveKalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)


class Track:
    def __init__(self, track_id, initial_detection, frame_idx):
        self.id = track_id
        self.points = {frame_idx: initial_detection}
        self.age = 0
        self.total_visible_count = 1
        self.last_frame_seen = frame_idx

    def get_ordered_points(self):
        return [self.points[fi] for fi in sorted(self.points.keys())]


class UltraTrack(Track):
    pass


class WaveParticleTrack(Track):
    pass


class StrictTrack(Track):
    pass


# ----------------- 轨迹匹配优化版本 -----------------

class TrajectoryMatcher:
    """优化的轨迹匹配器，专门针对波浪粒子运动"""

    def __init__(self, F_matrix, params):
        self.F = F_matrix
        self.params = params
        self.debug = params.get('debug', False)

    def match_trajectories(self, trajectories_left, trajectories_right):
        """主匹配函数"""
        # 1. 预处理和过滤
        valid_left, valid_right = self._preprocess_trajectories(
            trajectories_left, trajectories_right
        )

        if not valid_left or not valid_right:
            print("没有足够的有效轨迹进行匹配")
            return []

        print(f"\n过滤后: 左侧 {len(valid_left)} 条轨迹, 右侧 {len(valid_right)} 条轨迹")

        # 2. 构建成本矩阵
        cost_matrix = self._build_cost_matrix(valid_left, valid_right)

        # 3. 执行匹配
        matches = self._perform_matching(valid_left, valid_right, cost_matrix)

        # 4. 后处理验证
        final_matches = self._post_process_matches(matches)

        return final_matches

    def _preprocess_trajectories(self, trajs_left, trajs_right):
        """预处理轨迹，过滤短轨迹并提取有效信息"""
        min_len = self.params['min_traj_len']

        valid_left = []
        valid_right = []

        # 过滤并准备轨迹数据
        for traj in trajs_left:
            if len(traj.points) >= min_len:
                # 提取帧号和对应的点
                frames = sorted(traj.points.keys())
                points = [traj.points[f] for f in frames]
                valid_left.append({
                    'obj': traj,
                    'frames': frames,
                    'points': np.array(points),
                    'length': len(points),
                    'time_span': frames[-1] - frames[0]
                })

        for traj in trajs_right:
            if len(traj.points) >= min_len:
                frames = sorted(traj.points.keys())
                points = [traj.points[f] for f in frames]
                valid_right.append({
                    'obj': traj,
                    'frames': frames,
                    'points': np.array(points),
                    'length': len(points),
                    'time_span': frames[-1] - frames[0]
                })

        # 统计信息
        if valid_left:
            lengths = [t['length'] for t in valid_left]
            print(f"左侧轨迹长度: 最小={min(lengths)}, 最大={max(lengths)}, 平均={np.mean(lengths):.1f}")

        if valid_right:
            lengths = [t['length'] for t in valid_right]
            print(f"右侧轨迹长度: 最小={min(lengths)}, 最大={max(lengths)}, 平均={np.mean(lengths):.1f}")

        return valid_left, valid_right

    def _build_cost_matrix(self, valid_left, valid_right):
        """构建成本矩阵，使用多个指标"""
        n_left = len(valid_left)
        n_right = len(valid_right)
        cost_matrix = np.full((n_left, n_right), np.inf)

        print(f"\n构建 {n_left}x{n_right} 成本矩阵...")

        for i in range(n_left):
            for j in range(n_right):
                cost = self._compute_trajectory_cost(valid_left[i], valid_right[j])
                cost_matrix[i, j] = cost

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{n_left} 条左侧轨迹")

        return cost_matrix

    def _compute_trajectory_cost(self, traj_l, traj_r):
        """计算两条轨迹之间的匹配成本"""
        # 1. 时间重叠检查
        overlap_ratio = self._compute_temporal_overlap(traj_l, traj_r)
        if overlap_ratio < self.params['min_overlap_ratio']:
            return np.inf

        # 2. 获取共同时间段的轨迹
        common_traj_l, common_traj_r = self._extract_common_timespan(traj_l, traj_r)
        if common_traj_l is None or len(common_traj_l) < self.params['min_common_frames']:
            return np.inf

        # 3. 计算多个成本指标
        costs = []

        # 3.1 形状相似度（使用Procrustes分析）
        shape_cost = self._compute_shape_similarity(common_traj_l, common_traj_r)
        costs.append(shape_cost * self.params['weight_shape'])

        # 3.2 运动相关性
        motion_cost = self._compute_motion_correlation(common_traj_l, common_traj_r)
        costs.append(motion_cost * self.params['weight_motion'])

        # 3.3 对极几何误差
        epipolar_cost = self._compute_epipolar_cost(common_traj_l, common_traj_r)
        costs.append(epipolar_cost * self.params['weight_epipolar'])

        # 3.4 速度一致性
        velocity_cost = self._compute_velocity_consistency(common_traj_l, common_traj_r)
        costs.append(velocity_cost * self.params['weight_velocity'])

        # 综合成本
        total_cost = np.sum(costs)

        return total_cost

    def _compute_temporal_overlap(self, traj_l, traj_r):
        """计算时间重叠比例"""
        start_l, end_l = traj_l['frames'][0], traj_l['frames'][-1]
        start_r, end_r = traj_r['frames'][0], traj_r['frames'][-1]

        overlap_start = max(start_l, start_r)
        overlap_end = min(end_l, end_r)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_len = overlap_end - overlap_start + 1
        min_span = min(end_l - start_l + 1, end_r - start_r + 1)

        return overlap_len / min_span

    def _extract_common_timespan(self, traj_l, traj_r):
        """提取共同时间段的轨迹点"""
        # 找到共同的帧
        common_frames = sorted(set(traj_l['frames']) & set(traj_r['frames']))

        if len(common_frames) < 2:
            return None, None

        # 如果没有足够的共同帧，尝试插值
        if len(common_frames) < self.params['min_common_frames']:
            # 找到时间重叠区间
            start = max(traj_l['frames'][0], traj_r['frames'][0])
            end = min(traj_l['frames'][-1], traj_r['frames'][-1])

            if end - start < self.params['min_common_frames']:
                return None, None

            # 创建共同的时间点
            common_frames = list(range(start, end + 1))

            # 插值左侧轨迹
            points_l = self._interpolate_trajectory(
                traj_l['frames'], traj_l['points'], common_frames
            )

            # 插值右侧轨迹
            points_r = self._interpolate_trajectory(
                traj_r['frames'], traj_r['points'], common_frames
            )

            return points_l, points_r
        else:
            # 直接使用共同帧的点
            points_l = []
            points_r = []

            frame_to_point_l = dict(zip(traj_l['frames'], traj_l['points']))
            frame_to_point_r = dict(zip(traj_r['frames'], traj_r['points']))

            for f in common_frames:
                points_l.append(frame_to_point_l[f])
                points_r.append(frame_to_point_r[f])

            return np.array(points_l), np.array(points_r)

    def _interpolate_trajectory(self, frames, points, target_frames):
        """插值轨迹到目标帧"""
        if len(frames) < 2:
            return None

        # 分别对x和y坐标进行插值
        interp_x = interp1d(frames, points[:, 0], kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(frames, points[:, 1], kind='linear',
                            bounds_error=False, fill_value='extrapolate')

        # 插值到目标帧
        new_x = interp_x(target_frames)
        new_y = interp_y(target_frames)

        return np.column_stack([new_x, new_y])

    def _compute_shape_similarity(self, traj_l, traj_r):
        """计算形状相似度（归一化后）"""
        # 中心化
        traj_l_centered = traj_l - np.mean(traj_l, axis=0)
        traj_r_centered = traj_r - np.mean(traj_r, axis=0)

        # 归一化尺度
        scale_l = np.sqrt(np.sum(traj_l_centered ** 2))
        scale_r = np.sqrt(np.sum(traj_r_centered ** 2))

        if scale_l > 0 and scale_r > 0:
            traj_l_norm = traj_l_centered / scale_l
            traj_r_norm = traj_r_centered / scale_r

            # 计算形状差异
            diff = traj_l_norm - traj_r_norm
            shape_cost = np.sqrt(np.mean(diff ** 2))
        else:
            shape_cost = np.inf

        return shape_cost

    def _compute_motion_correlation(self, traj_l, traj_r):
        """计算运动相关性"""
        if len(traj_l) < 3:
            return np.inf

        # 计算速度
        vel_l = np.diff(traj_l, axis=0)
        vel_r = np.diff(traj_r, axis=0)

        # 计算速度大小的相关性
        speed_l = np.linalg.norm(vel_l, axis=1)
        speed_r = np.linalg.norm(vel_r, axis=1)

        if len(speed_l) > 1:
            # 归一化
            if np.std(speed_l) > 0 and np.std(speed_r) > 0:
                speed_l_norm = (speed_l - np.mean(speed_l)) / np.std(speed_l)
                speed_r_norm = (speed_r - np.mean(speed_r)) / np.std(speed_r)

                # 计算相关系数
                correlation = np.corrcoef(speed_l_norm, speed_r_norm)[0, 1]

                # 转换为成本（1 - correlation）
                motion_cost = 1 - correlation if not np.isnan(correlation) else 1.0
            else:
                motion_cost = 1.0
        else:
            motion_cost = 1.0

        return motion_cost

    def _compute_epipolar_cost(self, traj_l, traj_r):
        """计算对极几何误差"""
        n_points = min(len(traj_l), len(traj_r), 10)  # 最多检查10个点

        if n_points < 3:
            return np.inf

        # 均匀采样点
        indices = np.linspace(0, len(traj_l) - 1, n_points, dtype=int)

        errors = []
        for idx in indices:
            pt_l = np.array([traj_l[idx, 0], traj_l[idx, 1], 1.0])
            pt_r = np.array([traj_r[idx, 0], traj_r[idx, 1], 1.0])

            # 计算对极线
            line_r = self.F @ pt_l
            line_l = self.F.T @ pt_r

            # 计算点到对极线的距离
            dist_r = abs(pt_r.T @ line_r) / np.sqrt(line_r[0] ** 2 + line_r[1] ** 2)
            dist_l = abs(pt_l.T @ line_l) / np.sqrt(line_l[0] ** 2 + line_l[1] ** 2)

            errors.append((dist_r + dist_l) / 2)

        # 使用中位数而不是平均值，对异常值更鲁棒
        epipolar_cost = np.median(errors)

        return epipolar_cost

    def _compute_velocity_consistency(self, traj_l, traj_r):
        """计算速度一致性"""
        if len(traj_l) < 2:
            return np.inf

        # 计算速度向量
        vel_l = np.diff(traj_l, axis=0)
        vel_r = np.diff(traj_r, axis=0)

        # 计算速度差异
        vel_diff = vel_l - vel_r

        # 归一化
        mean_speed = (np.mean(np.linalg.norm(vel_l, axis=1)) +
                      np.mean(np.linalg.norm(vel_r, axis=1))) / 2

        if mean_speed > 0:
            normalized_diff = np.linalg.norm(vel_diff, axis=1) / mean_speed
            velocity_cost = np.mean(normalized_diff)
        else:
            velocity_cost = 0.0

        return velocity_cost

    def _perform_matching(self, valid_left, valid_right, cost_matrix):
        """执行匹配"""
        # 使用匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost < self.params['max_total_cost']:
                matches.append({
                    'left': valid_left[r],
                    'right': valid_right[c],
                    'cost': cost,
                    'left_idx': r,
                    'right_idx': c
                })

        print(f"\n初步匹配: 找到 {len(matches)} 对候选匹配")

        return matches

    def _post_process_matches(self, matches):
        """后处理验证匹配结果"""
        if not matches:
            return []

        # 按成本排序
        matches.sort(key=lambda x: x['cost'])

        # 额外验证
        final_matches = []
        used_left = set()
        used_right = set()

        for match in matches:
            # 确保没有重复使用
            if match['left_idx'] in used_left or match['right_idx'] in used_right:
                continue

            # 额外的验证
            if self._verify_match(match):
                final_matches.append((match['left']['obj'], match['right']['obj']))
                used_left.add(match['left_idx'])
                used_right.add(match['right_idx'])

                if self.debug:
                    print(f"接受匹配: 左轨迹{match['left']['obj'].id} <-> "
                          f"右轨迹{match['right']['obj'].id}, 成本={match['cost']:.3f}")

        print(f"\n最终匹配: {len(final_matches)} 对高质量匹配")

        return final_matches

    def _verify_match(self, match):
        """验证单个匹配"""
        # 可以添加额外的验证逻辑
        # 例如：检查3D重建后的轨迹是否合理
        return True


def visualize_matches(matched_pairs, img_left_path, img_right_path, output_path):
    """可视化匹配结果"""
    # 读取示例图像
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)

    if img_left is None or img_right is None:
        print("无法读取图像进行可视化")
        return

    # 拼接图像
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img_left
    vis[:h2, w1:w1 + w2] = img_right

    # 绘制匹配的轨迹
    colors = np.random.randint(0, 255, (len(matched_pairs), 3))

    for i, (traj_l, traj_r) in enumerate(matched_pairs[:20]):  # 最多显示20对
        color = tuple(map(int, colors[i]))

        # 获取轨迹点
        points_l = traj_l.get_ordered_points()
        points_r = traj_r.get_ordered_points()

        # 绘制左侧轨迹
        for j in range(1, len(points_l)):
            pt1 = tuple(map(int, points_l[j - 1]))
            pt2 = tuple(map(int, points_l[j]))
            cv2.line(vis, pt1, pt2, color, 2)

        # 绘制右侧轨迹（需要偏移）
        for j in range(1, len(points_r)):
            pt1 = tuple(map(int, (points_r[j - 1][0] + w1, points_r[j - 1][1])))
            pt2 = tuple(map(int, (points_r[j][0] + w1, points_r[j][1])))
            cv2.line(vis, pt1, pt2, color, 2)

        # 连接对应点
        if len(points_l) > 0 and len(points_r) > 0:
            pt_l = tuple(map(int, points_l[len(points_l) // 2]))
            pt_r = tuple(map(int, (points_r[len(points_r) // 2][0] + w1,
                                   points_r[len(points_r) // 2][1])))
            cv2.line(vis, pt_l, pt_r, color, 1)

    cv2.imwrite(output_path, vis)
    print(f"匹配可视化已保存到: {output_path}")


def run_optimized_matching(traj_file_left, traj_file_right, calib_params_file,
                           output_matched_file, matching_params):
    """运行优化的轨迹匹配"""

    # 加载数据
    try:
        with open(traj_file_left, 'rb') as f:
            trajectories_left = pickle.load(f)
        with open(traj_file_right, 'rb') as f:
            trajectories_right = pickle.load(f)

        calib_data = np.load(calib_params_file)
        F_matrix = calib_data['F']

        print(f"加载了 {len(trajectories_left)} 条左侧轨迹和 {len(trajectories_right)} 条右侧轨迹")

    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # 创建匹配器并执行匹配
    matcher = TrajectoryMatcher(F_matrix, matching_params)
    matched_pairs = matcher.match_trajectories(trajectories_left, trajectories_right)

    # 保存结果
    if matched_pairs:
        os.makedirs(os.path.dirname(output_matched_file), exist_ok=True)
        with open(output_matched_file, 'wb') as f:
            pickle.dump(matched_pairs, f)
        print(f"\n匹配结果已保存到: {output_matched_file}")

        # 可选：可视化
        if matching_params.get('visualize', False):
            vis_path = output_matched_file.replace('.pkl', '_vis.jpg')
            # 需要提供示例图像路径
            visualize_matches(matched_pairs, img_left_path, img_right_path, vis_path)
    else:
        print("\n未找到有效的匹配对")


if __name__ == '__main__':
    # 文件路径 - 根据你实际使用的轨迹文件调整
    # 可选: trajectories_2d_left.pkl, trajectories_2d_left_wave.pkl, trajectories_2d_left_strict.pkl 等
    traj_l_file = "../data/trajectories/trajectories_2d_left.pkl"
    traj_r_file = "../data/trajectories/trajectories_2d_right.pkl"
    calib_file = "../camera_calibration/params/stereo_calib_params_from_matlab_full.npz"
    out_matched_pkl = "../data/trajectories/matched_pairs_2d_optimized.pkl"

    # 优化的匹配参数 - 针对波浪粒子运动
    matching_parameters = {
        # 基本参数
        'min_traj_len': 20,  # 最小轨迹长度（确保轨迹质量）
        'min_common_frames': 15,  # 最少共同帧数
        'min_overlap_ratio': 0.5,  # 最小时间重叠比例

        # 成本权重（根据重要性调整）
        'weight_shape': 1.0,  # 形状相似度权重
        'weight_motion': 2.0,  # 运动相关性权重（重要）
        'weight_epipolar': 3.0,  # 对极几何权重（最重要）
        'weight_velocity': 1.0,  # 速度一致性权重

        # 阈值
        'max_total_cost': 10.0,  # 最大总成本阈值

        # 其他
        'debug': True,  # 调试模式
        'visualize': True  # 是否生成可视化
    }

    # 如果轨迹较短，可以调整参数
    # matching_parameters.update({
    #     'min_traj_len': 10,
    #     'min_common_frames': 8,
    #     'min_overlap_ratio': 0.3
    # })

    print("=== 优化的轨迹匹配系统 ===")
    print("\n参数设置:")
    for key, value in matching_parameters.items():
        print(f"  {key}: {value}")

    run_optimized_matching(
        traj_l_file, traj_r_file, calib_file,
        out_matched_pkl, matching_parameters
    )