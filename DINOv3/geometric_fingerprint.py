"""
几何指纹验证：用多个小圆片的相对位置关系作为特征
===================================================
基于特征缓存中的关键点坐标，不依赖 DINO 特征，
验证"小圆片星座"的区分度。

方法：
  1. Shape Context 描述子（log-polar 直方图）
  2. 邻域相对位置向量（KNN 几何指纹）
  3. 在真实数据上验证区分度，与 DINO 特征对比

输出：
  - 终端量化报告
  - 诊断图对比 DINO vs 几何指纹
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


# ============================================================
# 几何指纹方法 1：Shape Context 描述子
# ============================================================

def shape_context_descriptor(points, point_idx, n_radial=5, n_angular=12, r_inner=5, r_outer=300):
    """
    计算单个点的 Shape Context 描述子（Belongie et al. 2002）。

    原理：以该点为中心，在 log-polar 坐标系下统计周围点的分布。
    不同位置的"星座"会产生不同的直方图。

    Args:
        points: [N, 2] 所有关键点坐标
        point_idx: 目标点索引
        n_radial: 径向 bin 数量
        n_angular: 角度 bin 数量
        r_inner: 最小半径（排除自身）
        r_outer: 最大半径

    Returns:
        desc: [n_radial * n_angular] 归一化直方图
    """
    center = points[point_idx]
    rel = points - center  # [N, 2]
    dists = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
    angles = np.arctan2(rel[:, 1], rel[:, 0])  # [-π, π]

    # 排除自身
    mask = (dists > r_inner) & (dists < r_outer)
    if mask.sum() < 3:
        return np.zeros(n_radial * n_angular)

    dists_f = dists[mask]
    angles_f = angles[mask]

    # log-polar binning
    log_r = np.logspace(np.log10(r_inner), np.log10(r_outer), n_radial + 1)
    angle_bins = np.linspace(-np.pi, np.pi, n_angular + 1)

    desc = np.zeros((n_radial, n_angular))
    for i in range(n_radial):
        for j in range(n_angular):
            in_bin = (dists_f >= log_r[i]) & (dists_f < log_r[i + 1]) & \
                     (angles_f >= angle_bins[j]) & (angles_f < angle_bins[j + 1])
            desc[i, j] = in_bin.sum()

    desc_flat = desc.flatten()
    if desc_flat.sum() > 0:
        desc_flat = desc_flat / desc_flat.sum()
    return desc_flat


def compute_all_shape_contexts(points, **kwargs):
    """为所有点计算 Shape Context 描述子。"""
    N = points.shape[0]
    descs = np.zeros((N, kwargs.get('n_radial', 5) * kwargs.get('n_angular', 12)))
    for i in range(N):
        descs[i] = shape_context_descriptor(points, i, **kwargs)
    return descs


# ============================================================
# 几何指纹方法 2：KNN 相对位置向量
# ============================================================

def knn_relative_positions(points, K=8):
    """
    为每个点计算其 K 近邻的相对位置向量，作为几何指纹。

    相对位置向量包含方向和距离信息，不同位置的"星座"不同。

    Args:
        points: [N, 2] 关键点坐标
        K: 近邻数量

    Returns:
        fingerprint: [N, 2*K] 每个点的几何指纹（按角度排序）
    """
    N = points.shape[0]
    if N < K + 1:
        return np.zeros((N, 2 * K))

    dists = cdist(points, points)  # [N, N]
    # 排除自身
    np.fill_diagonal(dists, np.inf)
    knn_indices = np.argpartition(dists, K, axis=1)[:, :K]  # [N, K]

    fingerprints = np.zeros((N, 2 * K))
    for i in range(N):
        neighbors = points[knn_indices[i]]  # [K, 2]
        rel = neighbors - points[i]  # [K, 2]
        # 按角度排序，保证旋转不变性
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        sort_idx = np.argsort(angles)
        rel_sorted = rel[sort_idx]
        fingerprints[i] = rel_sorted.flatten()

    return fingerprints


# ============================================================
# 验证：几何指纹的区分度
# ============================================================

def analyze_fingerprint_discriminability(fingerprints, points, name="Fingerprint"):
    """
    分析几何指纹的区分度，用余弦相似度矩阵。
    与 DINO 特征的验证维度对齐。
    """
    N = fingerprints.shape[0]
    if N < 2:
        return None

    # 余弦相似度
    fp = torch.from_numpy(fingerprints).float()
    fp_norm = F.normalize(fp, dim=-1)
    sim = fp_norm @ fp_norm.T
    triu = sim[torch.triu(torch.ones_like(sim), diagonal=1).bool()]

    # 按空间距离分组
    spatial_dist = cdist(points, points)
    spatial_triu = spatial_dist[np.triu_indices_from(spatial_dist, k=1)]

    near_mask = spatial_triu < 100
    far_mask = spatial_triu > 500

    triu_np = triu.numpy()

    return {
        'name': name,
        'n_kp': N,
        'sim_mean': triu.mean().item(),
        'sim_std': triu.std().item(),
        'sim_lt_0.3': (triu < 0.3).float().mean().item(),
        'sim_lt_0.5': (triu < 0.5).float().mean().item(),
        'sim_gt_0.8': (triu > 0.8).float().mean().item(),
        'near_sim_mean': triu_np[near_mask].mean() if near_mask.any() else 0,
        'far_sim_mean': triu_np[far_mask].mean() if far_mask.any() else 0,
    }


# ============================================================
# 验证：几何指纹的跨视图一致性
# ============================================================

def analyze_fingerprint_cross_view(points_l, points_r, K=8):
    """
    左右图同一物理点的几何指纹应该相似。
    验证：极线对齐位置 vs 随机偏移位置。
    """
    fp_l = knn_relative_positions(points_l, K=K)
    fp_r = knn_relative_positions(points_r, K=K)

    fp_l_norm = F.normalize(torch.from_numpy(fp_l).float(), dim=-1)
    fp_r_norm = F.normalize(torch.from_numpy(fp_r).float(), dim=-1)

    N_l = points_l.shape[0]
    N_r = points_r.shape[0]

    aligned_sims = []
    shuffled_sims = []

    # 按极线行分组
    for y_bin in np.arange(0, 1600, 50):  # 50px 行带
        mask_l = (points_l[:, 1] >= y_bin) & (points_l[:, 1] < y_bin + 50)
        mask_r = (points_r[:, 1] >= y_bin) & (points_r[:, 1] < y_bin + 50)

        indices_l = np.where(mask_l)[0]
        indices_r = np.where(mask_r)[0]

        if len(indices_l) < 2 or len(indices_r) < 2:
            continue

        # 按 x 坐标排序后配对（近似匹配）
        sorted_l = indices_l[np.argsort(points_l[indices_l, 0])]
        sorted_r = indices_r[np.argsort(points_r[indices_r, 0])]

        n = min(len(sorted_l), len(sorted_r))
        for i in range(n):
            sim = (fp_l_norm[sorted_l[i]] * fp_r_norm[sorted_r[i]]).sum().item()
            aligned_sims.append(sim)

            # 随机配对
            j = np.random.randint(0, len(indices_r))
            sim_shuf = (fp_l_norm[sorted_l[i]] * fp_r_norm[indices_r[j]]).sum().item()
            shuffled_sims.append(sim_shuf)

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
# 主流程
# ============================================================

def main():
    cfg = Config()
    cache_dir = cfg.FEATURE_CACHE_DIR
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    if not cache_files:
        print("[错误] 未找到缓存文件")
        return

    # 采样 5 帧
    sample_files = cache_files[::max(1, len(cache_files) // 5)][:5]
    sample_files = list(set([cache_files[0], cache_files[-1]] + sample_files))
    sample_files.sort()

    print(f"特征缓存共 {len(cache_files)} 帧，采样 {len(sample_files)} 帧")

    # 累积统计
    sc_results = []   # Shape Context
    knn_results = []  # KNN 相对位置
    dino_results = [] # DINO 特征（对照组）
    cross_view_sc = []
    cross_view_knn = []

    for fpath in tqdm(sample_files, desc="分析"):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        c = torch.load(fpath, map_location='cpu', weights_only=False)
        kp_l = c['keypoints_left'].numpy()
        kp_r = c['keypoints_right'].numpy()
        feat_l = c['feat_left']

        # Shape Context 描述子
        descs_sc = compute_all_shape_contexts(kp_l)
        r = analyze_fingerprint_discriminability(descs_sc, kp_l, name="ShapeContext")
        if r:
            r['frame'] = fname
            sc_results.append(r)

        # KNN 相对位置指纹
        fp_knn = knn_relative_positions(kp_l, K=8)
        r = analyze_fingerprint_discriminability(fp_knn, kp_l, name="KNN-8")
        if r:
            r['frame'] = fname
            knn_results.append(r)

        # DINO 特征（对照组）
        C, Hf, Wf = feat_l.shape
        row = torch.round(c['keypoints_left'][:, 1] / 16).long()
        col = torch.round(c['keypoints_left'][:, 0] / 16).long()
        valid = (row >= 0) & (row < Hf) & (col >= 0) & (col < Wf)
        row, col = row[valid], col[valid]
        dino_feats = feat_l[:, row, col].T.float()  # [N, C]

        dino_sim = F.normalize(dino_feats, dim=-1) @ F.normalize(dino_feats, dim=-1).T
        triu = dino_sim[torch.triu(torch.ones_like(dino_sim), diagonal=1).bool()]
        dino_results.append({
            'name': 'DINOv3',
            'n_kp': dino_feats.shape[0],
            'sim_mean': triu.mean().item(),
            'sim_std': triu.std().item(),
            'sim_lt_0.5': (triu < 0.5).float().mean().item(),
            'sim_gt_0.8': (triu > 0.8).float().mean().item(),
            'frame': fname,
        })

        # 跨视图一致性
        cv = analyze_fingerprint_cross_view(kp_l, kp_r, K=8)
        if cv:
            cv['frame'] = fname
            cross_view_knn.append(cv)

        del c, feat_l, dino_feats
        gc.collect()

    # ============================================================
    # 汇总报告
    # ============================================================

    print("\n" + "=" * 70)
    print("  几何指纹 vs DINO 特征 —— 区分度对比")
    print("=" * 70)

    methods = [
        ("Shape Context (5×12=60维)", sc_results),
        ("KNN 相对位置 (K=8, 16维)", knn_results),
        ("DINOv3 特征 (768维)", dino_results),
    ]

    for name, results in methods:
        if not results:
            continue
        r = results[0]
        sim_lt_05 = np.array([x['sim_lt_0.5'] for x in results])
        sim_gt_08 = np.array([x['sim_gt_0.8'] for x in results])

        print(f"\n[{name}]")
        print(f"  相似度均值:        {r['sim_mean']:.3f} ± {r['sim_std']:.3f}")
        print(f"  相似度 < 0.5 占比: {sim_lt_05.mean():.1%} ± {sim_lt_05.std():.1%}")
        print(f"  相似度 > 0.8 占比: {sim_gt_08.mean():.1%} ± {sim_gt_08.std():.1%}")

        if 'near_sim_mean' in r:
            print(f"  近邻(<100px)相似度: {r['near_sim_mean']:.3f}")
            print(f"  远距(>500px)相似度: {r['far_sim_mean']:.3f}")

        if sim_lt_05.mean() > 0.5:
            print(f"  ✅ 超过半数对相似度 < 0.5，特征有区分度")
        else:
            print(f"  ⚠️  区分度不足")

    # 跨视图一致性
    if cross_view_knn:
        cv = cross_view_knn[0]
        gaps = np.array([x['gap'] for x in cross_view_knn])
        print(f"\n[KNN 几何指纹 — 跨视图一致性]")
        print(f"  对齐相似度:        {cv['aligned_mean']:.3f} ± {cv['aligned_std']:.3f}")
        print(f"  随机相似度:        {cv['shuffled_mean']:.3f} ± {cv['shuffled_std']:.3f}")
        print(f"  对齐-随机差距:     {cv['gap']:.4f} (多帧: {gaps.mean():.4f} ± {gaps.std():.4f})")
        if gaps.mean() > 0.05:
            print(f"  ✅ 跨视图一致性显著")
        else:
            print(f"  ⚠️  跨视图一致性弱")

    # ============================================================
    # 可视化对比
    # ============================================================

    out_dir = os.path.join(cache_dir, "validation")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Geometric Fingerprint vs DINO Feature Discriminability", fontsize=14)

    # 1. 相似度分布直方图（首帧对比）
    ax = axes[0, 0]
    labels = []
    values_lt05 = []
    values_gt08 = []
    colors = []

    for name, results in methods:
        if results:
            labels.append(name.split('(')[0].strip())
            values_lt05.append(results[0]['sim_lt_0.5'] * 100)
            values_gt08.append(results[0]['sim_gt_0.8'] * 100)
            colors.append(['#2ecc71', '#3498db', '#e74c3c'][len(labels) - 1])

    x = np.arange(len(labels))
    w = 0.3
    ax.bar(x - w/2, values_lt05, w, label='< 0.5 (good)', color='green', alpha=0.7)
    ax.bar(x + w/2, values_gt08, w, label='> 0.8 (bad)', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Pairwise Similarity Distribution (Frame 1)')
    ax.set_ylabel('% of pairs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2-4. 各方法多帧 < 0.5 占比
    for idx, (name, results) in enumerate(methods):
        if not results:
            continue
        ax = axes[(idx + 1) // 3, (idx + 1) % 3]
        sim_lt_05 = [r['sim_lt_0.5'] * 100 for r in results]
        sim_gt_08 = [r['sim_gt_0.8'] * 100 for r in results]
        frames = range(len(results))
        ax.bar(frames, sim_lt_05, label='<0.5', color='green', alpha=0.7)
        ax.bar(frames, sim_gt_08, label='>0.8', color='red', alpha=0.7)
        ax.set_title(name.split('(')[0].strip())
        ax.set_xlabel('Frame')
        ax.set_ylabel('%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 5. 跨视图一致性
    ax = axes[1, 1]
    if cross_view_knn:
        aligned = [x['aligned_mean'] for x in cross_view_knn]
        shuffled = [x['shuffled_mean'] for x in cross_view_knn]
        gaps = [x['gap'] for x in cross_view_knn]
        frames = range(len(cross_view_knn))
        ax.plot(frames, aligned, 'o-', label='Aligned', markersize=4, color='green')
        ax.plot(frames, shuffled, 's-', label='Shuffled', markersize=4, color='red')
        ax.plot(frames, gaps, '--', label='Gap', markersize=4, color='blue')
        ax.set_title('KNN Cross-View Consistency')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Cosine Similarity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 6. 总结
    ax = axes[1, 2]
    ax.axis('off')
    summary_lines = [
        "几何指纹 vs DINO 特征",
        "=" * 30,
        "",
        "理想特征：",
        "  • 自身小圆片 → 低相似度",
        "  • 近邻小圆片 → 中相似度",
        "  • 远距小圆片 → 低相似度",
        "  • 跨视图对齐 → 高相似度",
        "",
        "几何指纹优势：",
        "  • 不需要 GPU 推理",
        "  • 天然区分不同位置",
        "  • 可用已知点密度调参",
        "",
        "局限：",
        "  • 依赖小圆片检测精度",
        "  • 三维形变会导致失真",
    ]
    for i, line in enumerate(summary_lines):
        ax.text(0.05, 0.95 - i * 0.045, line, transform=ax.transAxes,
                fontsize=8, family='monospace', verticalalignment='top')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "geometric_vs_dino.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n诊断图: {out_path}")

    # ============================================================
    # 结论
    # ============================================================

    print(f"\n{'=' * 70}")
    print(f"  结论")
    print(f"{'=' * 70}")

    if sc_results and knn_results and dino_results:
        dino_lt05 = np.mean([r['sim_lt_0.5'] for r in dino_results])
        sc_lt05 = np.mean([r['sim_lt_0.5'] for r in sc_results])
        knn_lt05 = np.mean([r['sim_lt_0.5'] for r in knn_results])

        print(f"\n  区分度排名（相似度 < 0.5 占比，越高越好）：")
        ranked = sorted([
            ("DINOv3", dino_lt05),
            ("Shape Context", sc_lt05),
            ("KNN 相对位置", knn_lt05),
        ], key=lambda x: -x[1])

        for i, (name, val) in enumerate(ranked):
            print(f"    {i + 1}. {name}: {val:.1%}")

        best = ranked[0]
        if best[1] > 0.3:
            print(f"\n  ✅ {best[0]} 的区分度可用，建议作为匹配特征")
        else:
            print(f"\n  ⚠️  所有方法的区分度都不足 30%，建议增大 K（近邻数）或减小 r_outer")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()