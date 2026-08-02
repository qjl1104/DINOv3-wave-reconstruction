# particle_processing/02c_detection_v3_localmax.py
"""
v3 检测器：白顶帽 + 局部极大值 + 分水岭拆分 + 亚像素质心。直接作用于原始图像。

设计依据（2026-07-29 系列实验结论）：
  - 形状先验（圆度/凸度/惯性比）物理上不成立：泡沫圆片经相机斜视、局部侧倾
    成像为椭圆（轴比中位 0.68），粘连团更无形状可言；
    v1 严格形状过滤召回 ~31%，v2 放宽 ~48%（人工计数基准）。
  - 合格判据只剩：局部对比度高、尺寸在带内。单帧原型召回 ~90%。
  - 预处理右图被 CLAHE 放大纹理（01 系列脚本所致），左/右预处理非同代产物；
    本检测器用白顶帽(31px)取代背景建模，直接从原始 bmp 检测，完全可复现。

算法：
  1. 白顶帽（31px 椭圆核）= 原图 - 开运算：去除缓变背景/光照，保留泡沫尺度亮斑
  2. 自适应阈值 thr = max(THR_FLOOR, median + THR_K*MAD)
  3. 高斯平滑(σ=1.2) → min_dist 邻域局部极大值 > thr 为峰
  4. 亮掩膜内以峰为标记分水岭，粘连团拆成每峰一个盆地（只在多峰域 bbox 内做，提速）
  5. 盆地面积 ∈ [MIN_AREA, MAX_AREA]：下限杀噪声，上限杀水沫碎块
  6. 亚像素位置 = 盆地内 (响应-thr/2) 加权质心
  7. meta: (area, peak, contrast, big_comp)：big_comp=1 表示来自大连通域(水沫团)，
     供下游降权；不直接删除（水沫也跟随波面，由跟踪/匹配按行为裁决）
  8. 全序列静态点抑制（出现率>90% 的整数位置，如水池边角设备）

输出坐标系：矫正图坐标（与 canonical v1/v2/生产轨迹一致）。
  实现：每帧先按标定 remap 矫正（不做 CLAHE——预处理右图正是被它毁掉），
  再顶帽+检测。原始图直接检测会与生产链坐标系错位（中位 24px）。

输出 data/detections/detections_{side}_v3.pkl（与 v1/v2 同构 list[list[(x,y)]]）
   和 detections_{side}_v3_meta.pkl（每点 area/peak/contrast/big_comp）。
用法：../.venv_fs/Scripts/python.exe 02c_detection_v3_localmax.py [left|right|both] [帧数上限]
"""
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label as ndlabel, maximum_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")

TOPHAT_K = 31        # 白顶帽核（> 泡沫直径 ~16px，< 背景结构尺度）
SIGMA = 1.2          # 平滑核
MIN_DIST = 6         # 峰间最小距离 px
THR_FLOOR = 12.0     # 顶帽响应阈值下限（顶帽后噪声底 ~0-2）
THR_K = 8.0
MIN_AREA = 12        # 盆地面积下限 px²
MAX_AREA = 800       # 盆地面积上限 px²
BIG_COMP = 1500      # 连通域超过此面积视为水沫团（打 big_comp 标记）
STATIC_SUPPRESS = True

# 中间带 [BAND_LO, BIG_COMP] 水沫/粘连盘外观分类（2026-07-29 核验：LOO 90.5%）：
# 特征 [n_strong_peaks, edge_grad]，高斯朴素贝叶斯，42 个人工标注（左 f500）。
# 训练依据：水沫纹理多峰(中位20)且边缘弥散，标识物单/少峰(中位1)边缘清晰。
BAND_LO = 500
GNB_MARKER = ([4.414, 52.128], [5.84, 27.695])   # (mu, sd) n=29
GNB_FOAM = ([26.538, 52.904], [15.771, 14.469])  # (mu, sd) n=13

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_K, TOPHAT_K))


def band_is_foam(n_peaks, edge_grad):
    """中间带外观分类：True=水沫。GNB [n_peaks, edge_grad]。"""
    from scipy.stats import norm
    x = np.array([n_peaks, edge_grad])
    lp = []
    for mu, sd in (GNB_MARKER, GNB_FOAM):
        lp.append(np.log(0.5) + norm.logpdf(x, mu, sd).sum())
    return lp[1] > lp[0]


def detect_frame(img):
    """单帧（原始灰度图）检测 → (N,2) 亚像素坐标, (N,4) meta。"""
    th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, _KERNEL).astype(np.float32)
    med = np.median(th)
    mad = np.median(np.abs(th - med)) * 1.4826
    thr = max(THR_FLOOR, med + THR_K * mad)

    sm = gaussian_filter(th, SIGMA)
    peaks = (sm == maximum_filter(sm, size=MIN_DIST)) & (sm > thr)
    plab, n_pk = ndlabel(peaks)
    if n_pk == 0:
        return np.zeros((0, 2)), np.zeros((0, 4))

    mask = (sm > thr * 0.5).astype(np.uint8)
    clab, n_comp = ndlabel(mask)
    comp_of_peak = clab[plab > 0]
    n_per_comp = np.bincount(comp_of_peak, minlength=n_comp + 1)
    comp_area = np.bincount(clab.ravel(), minlength=n_comp + 1)
    # 边缘梯度图（中间带外观分类用，与训练同口径：顶帽图未平滑）
    gx = cv2.Sobel(th, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(th, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)

    pts, meta = [], []

    def emit(ys, xs, w, area, big):
        if area < MIN_AREA or area > MAX_AREA or w.sum() <= 0:
            return
        cx = float((xs * w).sum() / w.sum())
        cy = float((ys * w).sum() / w.sum())
        peak = float(sm[ys, xs].max())
        contrast = peak - float(np.median(sm[max(0, ys.min()-8):ys.max()+9,
                                             max(0, xs.min()-8):xs.max()+9]))
        pts.append((cx, cy))
        meta.append((area, peak, contrast, big))

    # 单峰连通域：直接质心
    single_comps = np.where(n_per_comp[1:] == 1)[0] + 1
    if len(single_comps):
        single_mask = np.isin(clab, single_comps)
        lab_s = clab[single_mask]
        ys, xs = np.where(single_mask)
        w = np.clip(sm[ys, xs] - thr * 0.5, 0, None)
        areas = np.bincount(lab_s, minlength=n_comp + 1)
        for cid in single_comps:
            sel = lab_s == cid
            emit(ys[sel], xs[sel], w[sel], int(areas[cid]),
                 1 if comp_area[cid] > BIG_COMP else 0)

    # 多峰连通域：bbox 内分水岭
    for cid in np.where(n_per_comp[1:] > 1)[0] + 1:
        ys, xs = np.where(clab == cid)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1
        # 面积标记 + 中间带外观分类（big=2 为带内水沫）
        if comp_area[cid] > BIG_COMP:
            big = 1
        elif comp_area[cid] >= BAND_LO:
            # 特征与训练时同口径：顶帽图(未平滑)上域内局部极大值(>0.5*域峰)数 + 边缘梯度
            pix_c = clab == cid
            vals = th[pix_c]
            sub = np.where(pix_c, th, 0)
            mf_c = maximum_filter(sub, size=6)
            pk_c = (sub == mf_c) & (sub > 0.5 * vals.max()) & pix_c
            _, n_strong = ndlabel(pk_c)
            dil = cv2.dilate(pix_c.astype(np.uint8), np.ones((6, 6), np.uint8)).astype(bool)
            ring = dil & ~pix_c
            edge_g = float(gmag[ring].mean()) if ring.any() else 0.0
            big = 2 if band_is_foam(float(n_strong), edge_g) else 0
        else:
            big = 0
        sub_markers = plab[y1:y2, x1:x2].astype(np.int32)
        sub_mask = ((clab[y1:y2, x1:x2] == cid) * 255).astype(np.uint8)
        cv2.watershed(cv2.cvtColor(sub_mask, cv2.COLOR_GRAY2BGR), sub_markers)
        for pk in np.unique(sub_markers[sub_markers > 0]):
            bys, bxs = np.where(sub_markers == pk)
            if len(bys) == 0:
                continue
            gys, gxs = bys + y1, bxs + x1
            w = np.clip(sm[gys, gxs] - thr * 0.5, 0, None)
            emit(gys, gxs, w, len(bys), big)

    return np.array(pts).reshape(-1, 2), np.array(meta).reshape(-1, 4)


def run_side(side, max_frames=None):
    import glob
    calib = np.load(CALIB)
    map1, map2 = calib[f"map1_{side}"], calib[f"map2_{side}"]
    files = sorted(glob.glob(os.path.join(ROOT, f"data/{side}_images/*.bmp")))
    if max_frames:
        files = files[:max_frames]
    dets, metas = [], []
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            dets.append([])
            metas.append(np.zeros((0, 4)))
            continue
        rect = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)  # 矫正系输出（兼容生产链）
        p, m = detect_frame(rect)
        dets.append([tuple(q) for q in p])
        metas.append(m)
        if (i + 1) % 200 == 0:
            print(f"  {side}: {i+1}/{len(files)}", flush=True)

    if STATIC_SUPPRESS and len(dets) > 50:
        from collections import Counter
        cnt = Counter()
        for d in dets:
            for (x, y) in set((round(x), round(y)) for x, y in d):
                cnt[(x, y)] += 1
        static_pts = {p for p, c in cnt.items() if c > 0.9 * len(dets)}
        if static_pts:
            for k in range(len(dets)):
                keep = [j for j, (x, y) in enumerate(dets[k])
                        if (round(x), round(y)) not in static_pts]
                dets[k] = [dets[k][j] for j in keep]
                metas[k] = metas[k][keep] if len(metas[k]) else metas[k]
            print(f"  {side}: 剔除静态点 {len(static_pts)} 个")

    counts = np.array([len(d) for d in dets])
    print(f"{side}: mean={counts.mean():.0f} min={counts.min()} max={counts.max()}")
    out = os.path.join(ROOT, f"data/detections/detections_{side}_v3.pkl")
    with open(out, "wb") as fp:
        pickle.dump(dets, fp)
    with open(out.replace("_v3.pkl", "_v3_meta.pkl"), "wb") as fp:
        pickle.dump(metas, fp)
    print(f"saved {out}")


if __name__ == "__main__":
    sides = sys.argv[1] if len(sys.argv) > 1 else "both"
    maxf = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for s in (["left", "right"] if sides == "both" else [sides]):
        run_side(s, maxf)
