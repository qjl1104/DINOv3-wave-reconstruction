# particle_processing/rematch_dino_common.py
"""
rematch_dino_v2.py（生产链）与 rematch_dino_assisted.py（已归档侧链）的共享逻辑：
DINO 描述子相似度打分、放宽视差波动门的扩展候选重扫、三角化质量过滤。
纯函数模块——pickle 桩类仍在各 rematch 脚本自身，勿移入本模块。
"""

import numpy as np

import rematch_rectified as rr

DINO_GATE = 0.55       # 描述子相似度门槛（真匹配 vs 错配分界，按分布调）
RELAX_DISP_STD = 60.0  # DINO 高时放宽的视差波动门（原 30）
MIN_NSIM = 10          # 参与接受判决的最少共同描述子帧数


def pair_dino_sim(dl, dr, common):
    """候选对共同帧上的描述子余弦相似度均值（两端都有描述子的帧）。"""
    sims = []
    for f in common:
        if f in dl and f in dr:
            a, b = dl[f], dr[f]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-6 and nb > 1e-6:
                sims.append(float(a @ b / (na * nb)))
    return (float(np.mean(sims)) if sims else -1.0), len(sims)


def score_candidates(candidates, rect_left, rect_right, td_l, td_r):
    """给几何候选对逐对打 DINO 分，返回 rows（dict 列表）。"""
    rows = []
    for i, j, st in candidates:
        fl, _ = rect_left[i]
        fr, _ = rect_right[j]
        common = sorted(set(fl) & set(fr))
        sim, nsim = pair_dino_sim(td_l[i], td_r[j], common)
        rows.append(dict(i=i, j=j, st=st, sim=sim, nsim=nsim))
    return rows


def extended_candidates(rect_left, rect_right, td_l, td_r):
    """扩展候选：dy 与视差范围满足、视差波动放宽到 RELAX_DISP_STD 的未入候选对
    （match_pairs 的 stats_grid 拿不到被 std 门拒的对，这里直接重算）。"""
    ext = []
    for i, (fl, pl) in enumerate(rect_left):
        fset_l = set(fl)
        for j, (fr, pr) in enumerate(rect_right):
            common = sorted(fset_l & set(fr))
            if len(common) < rr.MIN_OVERLAP:
                continue
            il = [fl.index(f) for f in common]
            ir = [fr.index(f) for f in common]
            dy = pl[il, 1] - pr[ir, 1]
            disp = pl[il, 0] - pr[ir, 0]
            med_dy, med_disp, std_disp = np.median(np.abs(dy)), np.median(disp), disp.std()
            if med_dy > rr.MAX_MED_DY:
                continue
            if not (rr.DISP_RANGE[0] <= med_disp <= rr.DISP_RANGE[1]):
                continue
            if std_disp <= rr.MAX_DISP_STD or std_disp > RELAX_DISP_STD:
                continue  # std ≤ MAX_DISP_STD 的已在几何候选里
            sim, nsim = pair_dino_sim(td_l[i], td_r[j], common)
            ext.append(dict(i=i, j=j, st=(len(common), med_dy, med_disp, std_disp),
                            sim=sim, nsim=nsim))
    return ext


def dino_accept(rows):
    """按 DINO 门槛拆分 rows → (接受, 拒绝)（nsim < MIN_NSIM 的两侧都不计）。"""
    acc = [r for r in rows if r["nsim"] >= MIN_NSIM and r["sim"] >= DINO_GATE]
    rej = [r for r in rows if r["nsim"] >= MIN_NSIM and r["sim"] < DINO_GATE]
    return acc, rej


def quality_filter(trajs_3d):
    """三角化质量过滤：中位深度范围 + 水平跨度上限（防 ID 串接）。"""
    kept = []
    for tr in trajs_3d:
        z_med = np.median(tr[:, 3])
        extent = max(np.ptp(tr[:, 1]), np.ptp(tr[:, 2]))
        if rr.DEPTH_RANGE[0] <= z_med <= rr.DEPTH_RANGE[1] and extent <= rr.MAX_EXTENT_UV:
            kept.append(tr)
    return kept
