"""
DINOv3 特征区分度验证脚本
===========================
基于特征缓存，不跑模型，多维度验证 DINOv3 特征在小圆片场景下的有效性。

验证维度：
  1. 背景 vs 小圆片区域的特征差异（特征是否"看到"了小圆片）
  2. 小圆片之间的特征区分度（不同小圆片的特征是否不同）
  3. 同列特征相似度 / 特征平滑性（左图关键点列 vs 右图同列——并非真实
     物理对应点，因为存在 250~1900px 视差；该维度检验的是特征沿极线
     方向的平滑性/平移不变性，而非跨视图对应关系）
  4. 密度-区分度关系（小圆片密度越高，特征是否越有区分度）
  5. 多帧统计稳定性

输出：
  - 终端打印量化报告
  - 保存 PNG 诊断图到 feature_cache/validation/
"""

import os
import sys
import glob
import gc

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config


def load_cache(cache_path):
    """加载缓存文件，返回特征图、关键点、灰度图。"""
    c = torch.load(cache_path, map_location='cpu', weights_only=False)
    feat_l = c['feat_left'].float()   # [C, Hf, Wf]
    feat_r = c['feat_right'].float()
    kp_l = c['keypoints_left']        # [N, 2]
    kp_r = c['keypoints_right']
    return feat_l, feat_r, kp_l, kp_r


def get_kp_feature_grid(feat, kp, patch_size=16):
    """将关键点映射到特征图网格，返回 (特征图行, 特征图列) 的整数索引。"""
    row = torch.round(kp[:, 1] / patch_size).long()
    col = torch.round(kp[:, 0] / patch_size).long()
    valid = (row >= 0) & (row < feat.shape[1]) & (col >= 0) & (col < feat.shape[2])
    return row[valid], col[valid], valid


def cosine_sim_matrix(features):
    """计算特征矩阵的余弦相似度矩阵 [N, N]"""
    f_norm = F.normalize(features, dim=-1)
    return f_norm @ f_norm.T


# ============================================================
# 维度 1：背景 vs 小圆片区域
# ============================================================

def analyze_background_vs_dots(feat_l, kp_l, patch_size=16):
    """
    比较小圆片所在特征位置与随机背景位置的特征差异。
    如果小圆片区域的特征与背景有显著差异，说明 DINO "看到"了小圆片。
    """
    C, Hf, Wf = feat_l.shape
    row, col, valid = get_kp_feature_grid(feat_l, kp_l, patch_size)

    # 小圆片区域特征
    kp_features = feat_l[:, row, col].T  # [N_kp, C]
    n_kp = kp_features.shape[0]

    # 随机背景位置（排除小圆片位置）
    kp_set = set(zip(row.tolist(), col.tolist()))
    all_positions = [(r, c) for r in range(Hf) for c in range(Wf) if (r, c) not in kp_set]
    np.random.seed(42)
    bg_indices = np.random.choice(len(all_positions), min(n_kp, len(all_positions)), replace=False)
    bg_positions = [all_positions[i] for i in bg_indices]
    bg_features = torch.stack([feat_l[:, r, c] for r, c in bg_positions])

    # 组内相似度：小圆片之间 vs 背景之间
    kp_sim = cosine_sim_matrix(kp_features)
    bg_sim = cosine_sim_matrix(bg_features)

    # 组间相似度：小圆片 vs 背景
    cross_sim = F.normalize(kp_features, dim=-1) @ F.normalize(bg_features, dim=-1).T

    # 取上三角（排除对角线）
    kp_triu = kp_sim[torch.triu(torch.ones_like(kp_sim), diagonal=1).bool()]
    bg_triu = bg_sim[torch.triu(torch.ones_like(bg_sim), diagonal=1).bool()]
    cross_flat = cross_sim.flatten()

    # 特征幅值
    kp_norm = kp_features.norm(dim=-1).mean().item()
    bg_norm = bg_features.norm(dim=-1).mean().item()

    return {
        'kp_intra_mean': kp_triu.mean().item(),
        'kp_intra_std': kp_triu.std().item(),
        'bg_intra_mean': bg_triu.mean().item(),
        'bg_intra_std': bg_triu.std().item(),
        'cross_mean': cross_flat.mean().item(),
        'cross_std': cross_flat.std().item(),
        'kp_norm': kp_norm,
        'bg_norm': bg_norm,
        'n_kp': n_kp,
    }


# ============================================================
# 维度 2：小圆片之间的特征区分度
# ============================================================

def analyze_dot_discriminability(feat_l, kp_l, patch_size=16):
    """
    分析不同小圆片之间的特征相似度分布。
    理想情况：大部分小圆片对的特征相似度低（< 0.5），
    只有少数近邻小圆片相似度高（因为周围上下文相似）。
    """
    row, col, valid = get_kp_feature_grid(feat_l, kp_l, patch_size)
    kp_features = feat_l[:, row, col].T  # [N, C]
    n_kp = kp_features.shape[0]

    if n_kp < 2:
        return None

    sim = cosine_sim_matrix(kp_features)
    triu = sim[torch.triu(torch.ones_like(sim), diagonal=1).bool()]

    # 按空间距离分组
    kp_np = kp_l[valid].numpy()
    spatial_dist = cdist(kp_np, kp_np)  # 像素距离
    spatial_triu = spatial_dist[np.triu_indices_from(spatial_dist, k=1)]

    # 近邻（< 100px）vs 远距离（> 500px）
    near_mask = spatial_triu < 100
    far_mask = spatial_triu > 500

    triu_np = triu.numpy()

    return {
        'n_kp': n_kp,
        'sim_mean': triu.mean().item(),
        'sim_std': triu.std().item(),
        'sim_lt_0.3': (triu < 0.3).float().mean().item(),
        'sim_lt_0.5': (triu < 0.5).float().mean().item(),
        'sim_gt_0.8': (triu > 0.8).float().mean().item(),
        'near_sim_mean': triu_np[near_mask].mean() if near_mask.any() else 0,
        'far_sim_mean': triu_np[far_mask].mean() if far_mask.any() else 0,
    }


# ============================================================
# 维度 3：跨视图一致性
# ============================================================

def analyze_cross_view_consistency(feat_l, feat_r, kp_l, kp_r, patch_size=16):
    """
    检验左图关键点所在列与右图同一列的特征相似度（同行同列）。

    注意：这里比较的是左右图的同一特征列，由于存在 250~1900px 的视差，
    同一列并不是同一个物理点。因此该维度实际检验的是特征沿极线方向的
    平滑性/平移不变性（同列相似度应高于随机列），而不是真实的跨视图
    对应关系——后者需要已知匹配真值才能验证。
    """
    row_l, col_l, valid_l = get_kp_feature_grid(feat_l, kp_l, patch_size)
    row_r, col_r, valid_r = get_kp_feature_grid(feat_r, kp_r, patch_size)

    C, Hf, Wf = feat_l.shape

    # 对所有左图关键点所在行，计算左右特征相关
    unique_rows = row_l.unique()
    aligned_sims = []
    shuffled_sims = []

    np.random.seed(42)
    for r in unique_rows.tolist():
        if r >= Hf:
            continue
        # 该行左图所有关键点位置的特征
        mask = row_l == r
        if mask.sum() < 2:
            continue
        cols_l_row = col_l[mask]
        feats_l_row = feat_l[:, r, cols_l_row].T  # [n, C]

        # 同一行右图特征
        feats_r_row = feat_r[:, r, :].T  # [Wf, C]

        # 同列对比：左图第 col 列 vs 右图第 col 列（同列而非真实匹配点，
        # 衡量特征沿极线的平滑性：相邻区域特征应比随机列更相似）
        feats_l_norm = F.normalize(feats_l_row, dim=-1)
        feats_r_norm = F.normalize(feats_r_row, dim=-1)

        # 取对角线（同列）
        cols_l_clamped = cols_l_row.clamp(0, Wf - 1)
        aligned = (feats_l_norm * feats_r_norm[cols_l_clamped]).sum(dim=-1)
        aligned_sims.extend(aligned.tolist())

        # 随机打乱列（破坏对应关系）
        shuffled_cols = np.random.permutation(Wf)[cols_l_clamped.numpy()]
        shuffled = (feats_l_norm * feats_r_norm[torch.tensor(shuffled_cols)]).sum(dim=-1)
        shuffled_sims.extend(shuffled.tolist())

    if not aligned_sims:
        return None

    aligned_arr = np.array(aligned_sims)
    shuffled_arr = np.array(shuffled_sims)

    return {
        'aligned_mean': aligned_arr.mean(),
        'aligned_std': aligned_arr.std(),
        'shuffled_mean': shuffled_arr.mean(),
        'shuffled_std': shuffled_arr.std(),
        'gap': aligned_arr.mean() - shuffled_arr.mean(),
        'n_pairs': len(aligned_arr),
    }


# ============================================================
# 维度 4：密度-区分度关系
# ============================================================

def analyze_density_discriminability(feat_l, kp_l, patch_size=16):
    """
    分析不同密度区域的特征区分度。
    将图像分成网格，统计每个网格的小圆片数量，
    区分高密度和低密度区域，对比特征相似度。
    """
    C, Hf, Wf = feat_l.shape
    row, col, valid = get_kp_feature_grid(feat_l, kp_l, patch_size)
    kp_features = feat_l[:, row, col].T  # [N, C]

    # 网格划分（特征图级别）：4×4 网格
    grid_h, grid_w = 4, 4
    cell_h = Hf // grid_h
    cell_w = Wf // grid_w

    densities = []
    cell_mean_sims = []  # 与 densities 一一对应的每格平均相似度
    high_density_sims = []
    low_density_sims = []

    # 第一遍：统计每个网格的密度与平均相似度
    for gh in range(grid_h):
        for gw in range(grid_w):
            mask = (row >= gh * cell_h) & (row < (gh + 1) * cell_h) & \
                   (col >= gw * cell_w) & (col < (gw + 1) * cell_w)
            n_in_cell = mask.sum().item()
            if n_in_cell < 3:
                continue
            densities.append(n_in_cell)

            cell_feats = kp_features[mask]
            sim = cosine_sim_matrix(cell_feats)
            triu = sim[torch.triu(torch.ones_like(sim), diagonal=1).bool()]
            cell_mean_sims.append(triu.mean().item())

    if not densities:
        return None

    # 第二遍：用全部网格密度的中位数（只算一次）分类高/低密度区
    density_median = np.median(densities)
    for n_in_cell, mean_sim in zip(densities, cell_mean_sims):
        if n_in_cell >= density_median:
            high_density_sims.append(mean_sim)
        else:
            low_density_sims.append(mean_sim)

    return {
        'n_cells': len(densities),
        'density_mean': np.mean(densities),
        'density_std': np.std(densities),
        'density_min': np.min(densities),
        'density_max': np.max(densities),
        'high_density_sim_mean': np.mean(high_density_sims) if high_density_sims else 0,
        'low_density_sim_mean': np.mean(low_density_sims) if low_density_sims else 0,
    }


# ============================================================
# 主流程
# ============================================================

def main():
    cfg = Config()
    cache_dir = cfg.FEATURE_CACHE_DIR
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    if not cache_files:
        print("[错误] 未找到缓存文件，请先运行 precompute_cache.py")
        return

    # 采样多帧（每 50 帧取 1 帧，最多 10 帧，避免内存爆炸）
    sample_files = cache_files[::max(1, len(cache_files) // 10)][:10]
    # 确保包含首尾帧
    sample_files = list(set([cache_files[0], cache_files[len(cache_files)//2], cache_files[-1]] + sample_files))
    sample_files.sort()

    print(f"特征缓存共 {len(cache_files)} 帧，采样 {len(sample_files)} 帧做分析")

    # 创建输出目录
    out_dir = os.path.join(cache_dir, "validation")
    os.makedirs(out_dir, exist_ok=True)

    # 累积统计
    all_dim1 = []
    all_dim2 = []
    all_dim3 = []
    all_dim4 = []

    for fpath in tqdm(sample_files, desc="分析特征"):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        feat_l, feat_r, kp_l, kp_r = load_cache(fpath)

        # 维度 1
        d1 = analyze_background_vs_dots(feat_l, kp_l)
        d1['frame'] = fname
        all_dim1.append(d1)

        # 维度 2
        d2 = analyze_dot_discriminability(feat_l, kp_l)
        if d2:
            d2['frame'] = fname
            all_dim2.append(d2)

        # 维度 3
        d3 = analyze_cross_view_consistency(feat_l, feat_r, kp_l, kp_r)
        if d3:
            d3['frame'] = fname
            all_dim3.append(d3)

        # 维度 4
        d4 = analyze_density_discriminability(feat_l, kp_l)
        if d4:
            d4['frame'] = fname
            all_dim4.append(d4)

        # 释放内存
        del feat_l, feat_r, kp_l, kp_r
        gc.collect()

    # ============================================================
    # 汇总报告
    # ============================================================

    print("\n" + "=" * 65)
    print("  DINOv3 特征区分度验证报告")
    print("=" * 65)

    # --- 维度 1：背景 vs 小圆片 ---
    if all_dim1:
        d1 = all_dim1[0]  # 首帧详情
        d1_arr = np.array([x['kp_intra_mean'] for x in all_dim1])
        print(f"\n[维度 1] 背景 vs 小圆片区域 ({len(all_dim1)} 帧)")
        print(f"  小圆片组内相似度:  {d1['kp_intra_mean']:.3f} ± {d1['kp_intra_std']:.3f}")
        print(f"  背景组内相似度:    {d1['bg_intra_mean']:.3f} ± {d1['bg_intra_std']:.3f}")
        print(f"  小圆片↔背景交叉:   {d1['cross_mean']:.3f} ± {d1['cross_std']:.3f}")
        print(f"  小圆片特征幅值:    {d1['kp_norm']:.1f}")
        print(f"  背景特征幅值:      {d1['bg_norm']:.1f}")
        if d1['kp_intra_mean'] > d1['cross_mean'] + 0.1:
            print(f"  ✅ 小圆片区域特征与背景有显著差异（组内 > 交叉 + 0.1）")
        else:
            print(f"  ❌ 小圆片区域特征与背景差异不足，可能特征退化")

    # --- 维度 2：小圆片区分度 ---
    if all_dim2:
        d2 = all_dim2[0]
        sim_lt_03_arr = np.array([x['sim_lt_0.3'] for x in all_dim2])
        sim_lt_05_arr = np.array([x['sim_lt_0.5'] for x in all_dim2])
        print(f"\n[维度 2] 小圆片之间的特征区分度 ({len(all_dim2)} 帧)")
        print(f"  首帧关键点数:      {d2['n_kp']}")
        print(f"  全对相似度均值:    {d2['sim_mean']:.3f} ± {d2['sim_std']:.3f}")
        print(f"  相似度 < 0.3 占比: {d2['sim_lt_0.3']:.1%} (多帧: {sim_lt_03_arr.mean():.1%} ± {sim_lt_03_arr.std():.1%})")
        print(f"  相似度 < 0.5 占比: {d2['sim_lt_0.5']:.1%} (多帧: {sim_lt_05_arr.mean():.1%} ± {sim_lt_05_arr.std():.1%})")
        print(f"  相似度 > 0.8 占比: {d2['sim_gt_0.8']:.1%}")
        print(f"  近邻(<100px)相似度: {d2['near_sim_mean']:.3f}")
        print(f"  远距(>500px)相似度: {d2['far_sim_mean']:.3f}")
        if sim_lt_05_arr.mean() > 0.5:
            print(f"  ✅ 超过半数小圆片对的特征相似度 < 0.5，特征有区分度")
        else:
            print(f"  ⚠️  多数小圆片对的特征相似度偏高，区分度不足")

    # --- 维度 3：同列特征相似度（特征平滑性/平移不变性，非真实对应） ---
    if all_dim3:
        d3 = all_dim3[0]
        gaps = np.array([x['gap'] for x in all_dim3])
        print(f"\n[维度 3] 同列特征相似度（特征平滑性/平移不变性，{len(all_dim3)} 帧）")
        print(f"  注意：同列 ≠ 同一物理点（存在 250~1900px 视差），不检验真实对应关系")
        print(f"  同列相似度:          {d3['aligned_mean']:.3f} ± {d3['aligned_std']:.3f}")
        print(f"  随机打乱相似度:      {d3['shuffled_mean']:.3f} ± {d3['shuffled_std']:.3f}")
        print(f"  同列-打乱差距:       {d3['gap']:.4f} (多帧: {gaps.mean():.4f} ± {gaps.std():.4f})")
        if gaps.mean() > 0.02:
            print(f"  ✅ 同列特征显著比随机列相似，特征沿极线方向平滑（平移不变性成立）")
        else:
            print(f"  ❌ 同列与随机打乱无显著差异，特征空间沿极线方向不平滑/判别过弱")

    # --- 维度 4：密度-区分度 ---
    if all_dim4:
        d4 = all_dim4[0]
        print(f"\n[维度 4] 密度-区分度关系 ({len(all_dim4)} 帧)")
        print(f"  网格数:            {d4['n_cells']}")
        print(f"  每格平均小圆片:    {d4['density_mean']:.1f} ± {d4['density_std']:.1f}")
        print(f"  密度范围:          [{d4['density_min']}, {d4['density_max']}]")
        print(f"  高密度区相似度:    {d4['high_density_sim_mean']:.3f}")
        print(f"  低密度区相似度:    {d4['low_density_sim_mean']:.3f}")
        if d4['high_density_sim_mean'] < d4['low_density_sim_mean']:
            print(f"  ✅ 高密度区相似度更低，说明密度越高特征越有区分度（符合预期）")
        else:
            print(f"  ⚠️  高密度区相似度反而更高，可能密度过高导致特征混淆")

    # ============================================================
    # 综合诊断
    # ============================================================

    print(f"\n{'=' * 65}")
    print(f"  综合诊断")
    print(f"{'=' * 65}")

    issues = []
    if all_dim1 and all_dim1[0]['kp_intra_mean'] <= all_dim1[0]['cross_mean'] + 0.05:
        issues.append("特征未有效区分小圆片和背景")
    if all_dim2 and all_dim2[0]['sim_lt_0.5'] < 0.3:
        issues.append("小圆片间特征区分度不足（<30% 对相似度 <0.5）")
    if all_dim3 and gaps.mean() < 0.01:
        issues.append("同列特征相似度与随机列无差异（特征沿极线方向不平滑，gap < 0.01）")
    if all_dim4 and all_dim4[0]['high_density_sim_mean'] >= all_dim4[0]['low_density_sim_mean']:
        issues.append("密度-区分度关系异常")

    if issues:
        for i, issue in enumerate(issues):
            print(f"  {i + 1}. {issue}")
    else:
        print(f"  ✅ 所有维度通过，DINOv3 特征在小圆片场景下有效")

    # ============================================================
    # 可视化
    # ============================================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"DINOv3 Feature Validation ({len(sample_files)} frames sampled)", fontsize=14)

    # 1. 小圆片 vs 背景相似度分布
    ax = axes[0, 0]
    if all_dim1:
        kp_vals = [d['kp_intra_mean'] for d in all_dim1]
        bg_vals = [d['bg_intra_mean'] for d in all_dim1]
        cross_vals = [d['cross_mean'] for d in all_dim1]
        frames = range(len(all_dim1))
        ax.plot(frames, kp_vals, 'o-', label='Dots intra', markersize=4)
        ax.plot(frames, bg_vals, 's-', label='BG intra', markersize=4)
        ax.plot(frames, cross_vals, 'x-', label='Cross', markersize=4)
        ax.set_title('Dim1: Dots vs Background')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Cosine Similarity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 2. 小圆片区分度直方图
    ax = axes[0, 1]
    if all_dim2:
        sim_lt_03 = [d['sim_lt_0.3'] * 100 for d in all_dim2]
        sim_lt_05 = [d['sim_lt_0.5'] * 100 for d in all_dim2]
        sim_gt_08 = [d['sim_gt_0.8'] * 100 for d in all_dim2]
        x = np.arange(len(all_dim2))
        w = 0.25
        ax.bar(x - w, sim_lt_03, w, label='<0.3', color='green', alpha=0.7)
        ax.bar(x, sim_lt_05, w, label='<0.5', color='blue', alpha=0.7)
        ax.bar(x + w, sim_gt_08, w, label='>0.8', color='red', alpha=0.7)
        ax.set_title('Dim2: Pairwise Similarity Distribution')
        ax.set_xlabel('Frame')
        ax.set_ylabel('% of pairs')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 3. 跨视图一致性
    ax = axes[0, 2]
    if all_dim3:
        aligned = [d['aligned_mean'] for d in all_dim3]
        shuffled = [d['shuffled_mean'] for d in all_dim3]
        gaps = [d['gap'] for d in all_dim3]
        frames = range(len(all_dim3))
        ax.plot(frames, aligned, 'o-', label='Aligned', markersize=4, color='green')
        ax.plot(frames, shuffled, 's-', label='Shuffled', markersize=4, color='red')
        ax.plot(frames, gaps, '--', label='Gap', markersize=4, color='blue')
        ax.set_title('Dim3: Cross-View Consistency')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Cosine Similarity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 4. 密度-区分度
    ax = axes[1, 0]
    if all_dim4:
        densities = [d['density_mean'] for d in all_dim4]
        high_sims = [d['high_density_sim_mean'] for d in all_dim4]
        low_sims = [d['low_density_sim_mean'] for d in all_dim4]
        frames = range(len(all_dim4))
        ax.plot(frames, high_sims, 'o-', label='High density sim', markersize=4, color='red')
        ax.plot(frames, low_sims, 's-', label='Low density sim', markersize=4, color='blue')
        ax.set_title('Dim4: Density vs Discriminability')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mean Similarity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 5. 特征幅值对比
    ax = axes[1, 1]
    if all_dim1:
        kp_norms = [d['kp_norm'] for d in all_dim1]
        bg_norms = [d['bg_norm'] for d in all_dim1]
        frames = range(len(all_dim1))
        ax.plot(frames, kp_norms, 'o-', label='Dots norm', markersize=4, color='orange')
        ax.plot(frames, bg_norms, 's-', label='BG norm', markersize=4, color='gray')
        ax.set_title('Feature Norms')
        ax.set_xlabel('Frame')
        ax.set_ylabel('L2 Norm')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 6. 关键点数量分布
    ax = axes[1, 2]
    if all_dim2:
        n_kps = [d['n_kp'] for d in all_dim2]
        frames = range(len(all_dim2))
        ax.bar(frames, n_kps, color='steelblue', alpha=0.7)
        ax.axhline(y=np.mean(n_kps), color='red', linestyle='--', label=f'Mean: {np.mean(n_kps):.0f}')
        ax.set_title('Keypoints per Frame')
        ax.set_xlabel('Frame')
        ax.set_ylabel('N keypoints')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "feature_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n诊断图已保存: {out_path}")

    print(f"\n{'=' * 65}")
    print(f"  验证完成")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()