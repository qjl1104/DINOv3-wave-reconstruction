# DINOv3/match_descriptor_nn.py
"""
修复1b：绕过训练头，直接用缓存的 DINOv3 特征做描述子最近邻匹配。

背景（诊断链）：
- 训练头的 soft-argmax 逐点预测位置 97% 不落在任何右图 blob ±8px 内
  → 粗匹配本身就不可靠，无法做"snap 到质心"细化（refine_subpixel.py 的结论）；
- 但 backbone 特征是冻结的基础模型输出、不依赖那个失败的训练头。
  经典用法：左 blob 描述子 vs 同极线带右 blob 描述子，
  余弦相似度 + 双向互查（cross-check）+ Lowe ratio 拒绝重复纹理歧义，
  双方都是 blob 质心（亚像素），再用 NCC 一维二次插值复核到亚像素。

与训练头方案的差别：训练头（Sinkhorn 全局分配 + soft-argmax 期望）会被
平滑先验拉向"平均面"（其自评波高缩水 58%、LR 一致性 0.15%）；
描述子 NN + 互查只对双方都互认的匹配放行，宁缺毋滥。

产出 pointclouds_dnn.pkl（frame → (N,3)），供 diag_dinov3_coherence.py 判决。
用法：../.venv_fs/Scripts/python.exe match_descriptor_nn.py
"""

import glob
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config  # noqa: E402
from refine_subpixel import ncc_1d  # noqa: E402
from utils import reproject_to_3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PKL = "pointclouds_dnn.pkl"
PATCH = 16          # DINOv3 ViT-B/16 特征步长
DY_MAX = 3.0        # 极线 |dy| 上限 px
RATIO = 1.05        # Lowe ratio：最佳相似度 / 次佳相似度 下限
SIM_MIN = 0.30      # 最佳相似度下限（L2 归一化余弦）
MIN_DISP = 10.0


def sample_desc(feat, kps):
    """feat: [C, Hf, Wf] fp16；kps: [N,2] 像素坐标 → 双线性采样描述子 [N, C]。"""
    C, Hf, Wf = feat.shape
    # 中心对齐映射：token j 覆盖像素 [16j, 16j+16)、中心 16j+8 → x/16-0.5，
    # 使 patch 中心的关键点恰好落在其 token 上（公式全仓库唯此一份，
    # compute_desc*.py / track_descriptor.py 均 import 本函数）
    gx = (kps[:, 0] / PATCH - 0.5).clamp(0, Wf - 1.001)
    gy = (kps[:, 1] / PATCH - 0.5).clamp(0, Hf - 1.001)
    x0, y0 = gx.floor().long(), gy.floor().long()
    x1, y1 = (x0 + 1).clamp(max=Wf - 1), (y0 + 1).clamp(max=Hf - 1)
    fx, fy = (gx - x0.float()).float(), (gy - y0.float()).float()
    f = feat.float()
    d = (f[:, y0, x0] * (1 - fx) * (1 - fy) + f[:, y1, x0] * (1 - fx) * fy +
         f[:, y0, x1] * fx * (1 - fy) + f[:, y1, x1] * fx * fy).T
    return F.normalize(d, dim=-1)


def main():
    cfg = Config()
    Q = np.load(cfg.CALIBRATION_FILE)["Q"]
    # 锚定脚本目录：CWD 不对时 glob 会落空，空结果会把共享输出 pkl 覆盖成空
    files = sorted(glob.glob(os.path.join(HERE, "feature_cache/left*.pt")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if not files:
        print(f"[错误] 未找到 {os.path.join(HERE, 'feature_cache/left*.pt')}，"
              "请先运行 precompute_cache.py 生成特征缓存")
        sys.exit(1)
    print(f"缓存帧数: {len(files)}")

    clouds = {}
    n_l, n_cand, n_pass = [], [], []
    sims = []
    t0 = time.time()
    for fi, fp in enumerate(files):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        kpl = d["keypoints_left"].float()
        kpr = d["keypoints_right"].float()
        if len(kpr) < 2:
            # 右点 <2 时 sim.topk(2) 直接 RuntimeError，跳过本帧
            print(f"[跳过] 帧 {fi}: 右关键点仅 {len(kpr)} 个，无法 topk(2)/ratio")
            clouds[fi] = np.zeros((0, 3))
            n_l.append(len(kpl))
            n_cand.append(0)
            n_pass.append(0)
            continue
        dl = sample_desc(d["feat_left"], kpl)    # [Nl, C]
        dr = sample_desc(d["feat_right"], kpr)   # [Nr, C]
        sim = dl @ dr.T                          # [Nl, Nr]

        # 极线带掩码：|dy| < DY_MAX
        dy_ok = (kpl[:, 1:2] - kpr[:, 1].unsqueeze(0)).abs() < DY_MAX
        sim = sim.masked_fill(~dy_ok, -1.0)

        # 正向：每个左点最佳/次佳右点
        v1, i1 = sim.topk(2, dim=1)
        best_r, sim1, sim2 = i1[:, 0], v1[:, 0], v1[:, 1]
        # 反向互查：右点的最佳左点是本左点
        best_l = sim.argmax(dim=0)
        mutual = best_l[best_r] == torch.arange(len(kpl))

        # 注：极线掩码后若只剩 1 个候选，sim2=-1 被 clamp 成 1e-6，ratio 检验
        # 形同虚设——此时靠 SIM_MIN 与互查兜底（仅备注，不改逻辑）
        ok = mutual & (sim1 > SIM_MIN) & (sim1 > RATIO * sim2.clamp(min=1e-6))
        idx = ok.nonzero(as_tuple=True)[0]

        kp_l_m, disp_m = [], []
        lgray, rgray = d["left_gray"].numpy(), d["right_gray"].numpy()
        for i in idx.tolist():
            j = best_r[i].item()
            xl, yl = kpl[i].tolist()
            xr, yr = kpr[j].tolist()
            ref = ncc_1d(lgray, rgray, int(round(xl)), int(round(yl)),
                         int(round(xr)), int(round(yr)))
            if ref is None:
                continue
            off, nccs = ref
            disp = xl - (xr + off)
            if disp > MIN_DISP:
                kp_l_m.append((xl, yl))
                disp_m.append(disp)
                sims.append(sim1[i].item())

        n_l.append(len(kpl))
        n_cand.append(len(idx))
        n_pass.append(len(disp_m))
        pts = np.zeros((0, 3))
        if len(disp_m) >= 10:
            pts = reproject_to_3d(np.array(kp_l_m), np.array(disp_m), Q).astype(np.float64)
            Z = pts[:, 2]
            pts = pts[(Z > 2000) & (Z < 15000)]
            if len(pts) >= 10:
                zm = np.median(pts[:, 2])
                ziqr = np.percentile(pts[:, 2], 75) - np.percentile(pts[:, 2], 25)
                pts = pts[np.abs(pts[:, 2] - zm) < 2.0 * max(ziqr, 1e-6)]
        clouds[fi] = pts
        if (fi + 1) % 200 == 0:
            print(f"{fi + 1}/{len(files)}  用时 {time.time() - t0:.0f}s  "
                  f"互查 {np.mean(n_cand[-200:]):.0f} NCC过 {np.mean(n_pass[-200:]):.0f}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(clouds, f)
    sims = np.array(sims)
    print(f"[输出] {OUT_PKL} | 总点数 {sum(len(v) for v in clouds.values())}")
    print(f"[统计] 左点/帧 {np.mean(n_l):.0f} → 互查通过 {np.mean(n_cand):.1f} "
          f"→ NCC+视差过滤 {np.mean(n_pass):.1f} | 最佳相似度中位 {np.median(sims):.3f}")


if __name__ == "__main__":
    main()
