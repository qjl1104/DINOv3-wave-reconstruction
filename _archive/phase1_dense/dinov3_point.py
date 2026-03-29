# -*- coding: utf-8 -*-
"""
Sparse stereo depth (white dots only) using DINOv3 features.
- Detect bright dots (binarize + CC + subpixel centroid)
- Extract ViT-B/16 (DINOv3) features with tiling (stride=16)
- For each left-dot, scan along the epipolar row on right feature map
  by cosine similarity; quadratic subpixel refine on the similarity curve
- Optional left-right cross-check
- disparity(px) -> depth(m): Z = fx * B / d  (fx pixels, B meters)
- Save CSV per frame + overlay; optional IDW preview (for visualization only)

Author: you + GPT-5 Thinking
"""

import os, glob, time, math, json, re
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ---- (optional) for fast IDW preview ----
try:
    from scipy.spatial import cKDTree

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ===================== Config =====================
class Config:
    # --- 数据目录 ---
    # !!!重要!!!: 左右目录中的图片必须有完全相同的文件名才能配对
    # 例如: lresult/frame001.png 和 rresult/frame001.png
    LEFT_DIR = r"D:\Research\wave_reconstruction_project\data\lresult"
    RIGHT_DIR = r"D:\Research\wave_reconstruction_project\data\rresult"
    OUT_ROOT = r"./experiments/dino_sparse_dots"

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # --- DINOv3 ---
    DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PATCH_SIZE = 16
    FEAT_NORM = True
    USE_CLAHE = True
    CLAHE_CLIP = 2.0
    CLAHE_GRID = (8, 8)

    # --- Tiling for full-res features ---
    TILE_H = 960
    TILE_W = 1024
    TILE_OVERLAP = 128

    # --- 相机标定参数 (优先使用 yaml, 否则手动填写) ---
    STEREO_YAML = r""
    FOCAL_LENGTH = 1800.0  # 像素 fx
    BASELINE_M = 0.20  # 米 B

    # --- 白点检测 ---
    DOT_THRESH = 220
    DOT_AREA_MIN = 3
    DOT_AREA_MAX = 500
    MEDIAN_BLUR = 0
    OPEN_KSIZE = 0

    # --- 特征匹配 (步长 = 16 像素) ---
    FEAT_POOL_RAD = 1
    MAX_DISP_PX = 120
    ROW_TOL_PX = 2
    SIM_MIN = 0.5
    SUBPIX = True
    CROSS_CHECK = True

    # --- IDW 可视化预览 ---
    MAKE_IDW_PREVIEW = True
    IDW_K_NEIGHBORS = 6
    IDW_POWER = 2.0
    IDW_DOWNSAMPLE = 8


cfg = Config()


# ===================== Utils: IO & basic =====================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def list_pairs(ldir, rdir, exts):
    """
    通过文件名末尾的数字后缀来查找并配对左右目录中的图像。
    例如: "lresult0001.png" 和 "rresult0001.png" 会被配对。
    """

    def collect(d):
        paths_map = {}  # key: 数字后缀, value: 完整路径
        names_map = {}  # key: 数字后缀, value: 文件名
        for ext in exts:
            for f in glob.glob(os.path.join(d, f"*{ext}")):
                basename = os.path.basename(f)
                # 使用正则表达式从文件名末尾提取数字
                match = re.search(r'(\d+)' + re.escape(ext) + '$', basename, re.IGNORECASE)
                if match:
                    numeric_suffix = match.group(1)
                    paths_map[numeric_suffix] = f
                    names_map[numeric_suffix] = basename
        return paths_map, names_map

    L_paths, L_names = collect(ldir)
    R_paths, _ = collect(rdir)  # 右侧只需要路径映射

    # ---- [调试代码] 检查找到的文件 ----
    print("-" * 50)
    print(f"[Debug] 正在检查目录...")
    print(f"[Debug] 左目录路径: {ldir}")
    print(f"[Debug] 在左目录中找到 {len(L_paths)} 个有效文件 (基于数字后缀)。")
    if len(L_paths) > 0:
        print(f"      示例数字后缀: {list(L_paths.keys())[:5]}")

    print(f"[Debug] 右目录路径: {rdir}")
    print(f"[Debug] 在右目录中找到 {len(R_paths)} 个有效文件 (基于数字后缀)。")
    if len(R_paths) > 0:
        print(f"      示例数字后缀: {list(R_paths.keys())[:5]}")
    # ---- [结束调试] ----

    # 找出左右目录中共有的数字后缀
    common_suffixes = sorted(set(L_paths.keys()) & set(R_paths.keys()))

    # ---- [调试代码] 检查匹配的后缀 ----
    print(f"[Debug] 找到 {len(common_suffixes)} 个共有的数字后缀（即有效图像对）。")
    if len(common_suffixes) > 0:
        print(f"      示例匹配后缀: {common_suffixes[:5]}")
    print("-" * 50)
    # ---- [结束调试] ----

    # 构建图像对列表，格式为 (文件名, 左图路径, 右图路径)
    return [(L_names[suffix], L_paths[suffix], R_paths[suffix]) for suffix in common_suffixes]


def read_stereo_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开立体标定yaml文件: {path}")
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    fs.release()
    if P1 is None or P2 is None:
        raise ValueError("YAML 文件必须包含 P1 和 P2 (矫正后的投影矩阵).")
    fx = float(P1[0, 0])
    B = float(-P2[0, 3] / fx)
    return fx, B


# ===================== White dot detector =====================
def detect_white_dots(gray_u8, thr, area_min, area_max, median_ksize=0, open_ksize=0):
    img = gray_u8
    if median_ksize and median_ksize >= 3:
        img = cv2.medianBlur(img, median_ksize)
    bw = (img >= thr).astype(np.uint8) * 255
    if open_ksize and open_ksize >= 3:
        kernel = np.ones((open_ksize, open_ksize), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    num, labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    ys, xs = [], []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area_min <= area <= area_max:
            cx, cy = cent[i]
            r = 3
            x0, y0 = int(round(cx)), int(round(cy))
            H, W = gray_u8.shape
            x1, x2 = max(0, x0 - r), min(W - 1, x0 + r)
            y1, y2 = max(0, y0 - r), min(H - 1, y0 + r)
            win = gray_u8[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
            if win.size >= 9 and win.sum() > 1e-3:
                yy, xx = np.mgrid[y1:y2 + 1, x1:x2 + 1]
                cy_ref = (win * yy).sum() / win.sum()
                cx_ref = (win * xx).sum() / win.sum()
                ys.append(float(cy_ref));
                xs.append(float(cx_ref))
            else:
                ys.append(float(cy));
                xs.append(float(cx))
    return np.array(ys, np.float32), np.array(xs, np.float32)


# ===================== DINO feature extractor (tiled) =====================
class DINOExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        self.patch = cfg.PATCH_SIZE

        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(cfg.DINO_MODEL, trust_remote_code=True).to(self.device)
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.patch * 2, self.patch * 2, device=self.device)
            out = self.model(pixel_values=dummy, output_hidden_states=True)
            hs = out.last_hidden_state.shape[-1]
        self.hidden = int(hs)

    def _prep_rgb(self, gray_u8):
        if self.cfg.USE_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=self.cfg.CLAHE_CLIP, tileGridSize=self.cfg.CLAHE_GRID)
            eq = clahe.apply(gray_u8)
            rgb = np.stack([eq] * 3, axis=-1).astype(np.float32) / 255.0
        else:
            rgb = np.stack([gray_u8] * 3, axis=-1).astype(np.float32) / 255.0
        ten = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        ten = (ten - self.mean) / self.std
        return ten

    def _tile_params(self, H, W):
        ps = self.patch
        th = max(ps, (self.cfg.TILE_H // ps) * ps)
        tw = max(ps, (self.cfg.TILE_W // ps) * ps)
        ov = (self.cfg.TILE_OVERLAP // ps) * ps
        th = min(th, ((H + ps - 1) // ps) * ps)
        tw = min(tw, ((W + ps - 1) // ps) * ps)
        return th, tw, ov

    @torch.no_grad()
    def extract_grid(self, gray_u8):
        H, W = gray_u8.shape
        ps = self.patch
        th, tw, ov = self._tile_params(H, W)

        Hf, Wf = (H + ps - 1) // ps, (W + ps - 1) // ps
        feat_acc = torch.zeros(self.hidden, Hf, Wf, device=self.device)
        w_acc = torch.zeros(1, Hf, Wf, device=self.device)

        y = 0
        while y < H:
            y1 = y
            y2 = min(y + th, H)
            if y2 - y1 < th and y1 > 0:
                y1 = max(0, y2 - th)
            x = 0
            while x < W:
                x1 = x
                x2 = min(x + tw, W)
                if x2 - x1 < tw and x1 > 0:
                    x1 = max(0, x2 - tw)

                tile = gray_u8[y1:y2, x1:x2]
                inp = self._prep_rgb(tile)
                out = self.model(pixel_values=inp, output_hidden_states=True)
                tokens = out.last_hidden_state  # [1, N_total, C]

                # 首先计算期望的patch token数量
                fh, fw = (y2 - y1) // ps, (x2 - x1) // ps
                num_patch_tokens = fh * fw

                # DINOv2/v3 模型在序列开头有额外的token（CLS, register tokens）
                # 我们只取序列末尾的 patch tokens
                # 这种方法比简单地检查 `+1` 更鲁棒
                patch_tokens = tokens[:, -num_patch_tokens:, :]

                C = patch_tokens.shape[-1]
                # 现在 reshape 的维度和数据量应该能匹配了
                feat = patch_tokens.reshape(1, fh, fw, C).permute(0, 3, 1, 2).squeeze(0)

                ys = (y1) // ps
                xs = (x1) // ps
                feat_acc[:, ys:ys + fh, xs:xs + fw] += feat
                w_acc[:, ys:ys + fh, xs:xs + fw] += 1.0

                if x + tw >= W: break
                x = x + tw - ov
            if y + th >= H: break
            y = y + th - ov

        feat = feat_acc / (w_acc + 1e-6)
        if self.cfg.FEAT_NORM:
            C, Hf, Wf = feat.shape
            feat2 = feat.permute(1, 2, 0).reshape(-1, C)
            feat2 = F.normalize(feat2, dim=1)
            feat = feat2.reshape(Hf, Wf, C).permute(2, 0, 1).contiguous()
        return feat.detach().cpu().float(), (Hf, Wf)


# ===================== Matching on feature grid =====================
def pool_feat(feat_C_HW, y_f, x_f, rad=1):
    C, Hf, Wf = feat_C_HW.shape
    y0 = max(0, y_f - rad);
    y1 = min(Hf - 1, y_f + rad)
    x0 = max(0, x_f - rad);
    x1 = min(Wf - 1, x_f + rad)
    patch = feat_C_HW[:, y0:y1 + 1, x0:x1 + 1].reshape(C, -1)
    v = patch.mean(dim=1)
    v = F.normalize(v, dim=0)
    return v


def scan_right_along_row(featL, featR, y_f, x_f, max_disp_f, sim_min=0.5, subpix=True):
    C, Hf, Wf = featL.shape
    if not (0 <= y_f < Hf and 0 <= x_f < Wf):
        return None
    tpl = pool_feat(featL, y_f, x_f, rad=cfg.FEAT_POOL_RAD)

    sims, xs = [], []
    for d in range(1, max_disp_f + 1):
        xr = x_f - d
        if xr < 0: break
        cand = pool_feat(featR, y_f, xr, rad=0)
        s = float(torch.dot(tpl, cand).clamp(-1, 1))
        sims.append(s);
        xs.append(xr)
    if len(sims) < 3:
        return None
    sims = np.array(sims, np.float32)
    imax = int(np.argmax(sims))
    smax = float(sims[imax])
    if smax < sim_min:
        return None
    d_int = imax + 1
    d = float(d_int)

    if subpix and 1 <= imax < len(sims) - 1:
        c_m1, c0, c_p1 = float(sims[imax - 1]), float(sims[imax]), float(sims[imax + 1])
        denom = (c_m1 - 2 * c0 + c_p1)
        if abs(denom) > 1e-6:
            delta = 0.5 * (c_m1 - c_p1) / denom
            d = d_int + delta

    xr_hat = x_f - d
    return d, xr_hat, smax


def cross_check(featR, featL, y_f, xr_f, xl_hint_f, max_disp_f, sim_min=0.5, tol=1.0):
    C, Hf, Wf = featR.shape
    if not (0 <= y_f < Hf and 0 <= xr_f < Wf):
        return False
    tpl = pool_feat(featR, y_f, int(round(xr_f)), rad=cfg.FEAT_POOL_RAD)
    sims, xs = [], []
    for d in range(1, max_disp_f + 1):
        xl = int(round(xr_f + d))
        if xl >= Wf: break
        cand = pool_feat(featL, y_f, xl, rad=0)
        s = float(torch.dot(tpl, cand).clamp(-1, 1))
        sims.append(s);
        xs.append(xl)
    if not sims:
        return False
    imax = int(np.argmax(sims))
    smax = float(sims[imax])
    if smax < sim_min:
        return False
    xl_hat = float(xs[imax])
    return abs(xl_hat - xl_hint_f) <= tol


# ===================== Depth utils/vis =====================
def disparity_to_depth(disp_px, fx, baseline_m):
    with np.errstate(divide='ignore', invalid='ignore'):
        z = fx * baseline_m / np.maximum(disp_px, 1e-6)
    return z


def colorize_depth_on_image(gray_u8, ys, xs, depths, vmin=None, vmax=None):
    img_color = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    if len(depths) == 0: return img_color
    d = np.array(depths, np.float32)
    valid_d = d[np.isfinite(d) & (d > 0)]
    if len(valid_d) == 0: return img_color

    if vmin is None: vmin = np.percentile(valid_d, 5)
    if vmax is None: vmax = np.percentile(valid_d, 95)
    vmax = max(vmax, vmin + 1e-6)

    norm = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    norm[~np.isfinite(d)] = 0  # handle non-finite values for color mapping
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    for i in range(len(ys)):
        if np.isfinite(depths[i]):
            c = cm[i].flatten()
            cv2.circle(img_color, (int(xs[i]), int(ys[i])), 3, tuple(int(v) for v in c), -1, lineType=cv2.LINE_AA)
    return img_color


def idw_preview(H, W, ys, xs, vals, ds=8, k=6, p=2.0):
    if not SCIPY_OK or len(vals) == 0: return None
    ys, xs = ys.astype(np.float32), xs.astype(np.float32)
    pts = np.stack([ys, xs], axis=1)
    tree = cKDTree(pts)
    grid_y, grid_x = np.mgrid[0:H:ds, 0:W:ds]
    coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)
    dists, idxs = tree.query(coords, k=min(k, len(pts)))
    dists = np.maximum(dists, 1e-6)
    w = 1.0 / (dists ** p)
    v = vals[idxs]
    out_small = (w * v).sum(axis=1) / (w.sum(axis=1) + 1e-6)
    out_small = out_small.reshape(grid_y.shape).astype(np.float32)
    out_full = cv2.resize(out_small, (W, H), interpolation=cv2.INTER_CUBIC)
    return out_full


def save_csv(path, rows, header=("y", "x", "disp_px", "depth_m", "sim")):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(header)
        for r in rows: w.writerow(r)


# ===================== Main =====================
def main():
    tag = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ensure_dir(os.path.join(cfg.OUT_ROOT, f"dino_sparse_{tag}"))
    out_vis = ensure_dir(os.path.join(out_dir, "vis"))
    out_csv = ensure_dir(os.path.join(out_dir, "csv"))

    if cfg.STEREO_YAML and os.path.isfile(cfg.STEREO_YAML):
        fx, B = read_stereo_yaml(cfg.STEREO_YAML)
        src = "yaml"
    else:
        fx, B = float(cfg.FOCAL_LENGTH), float(cfg.BASELINE_M)
        src = "manual"
    print(f"[信息] 相机参数: fx={fx:.3f}, B={B:.6f} m (来源={src})")

    pairs = list_pairs(cfg.LEFT_DIR, cfg.RIGHT_DIR, cfg.IMG_EXTS)
    if not pairs:
        print("\n[错误] 未找到任何可以配对的立体图像。")
        print("请检查以下几点：")
        print(f"1. 左目录路径是否正确且存在: '{cfg.LEFT_DIR}'")
        print(f"2. 右目录路径是否正确且存在: '{cfg.RIGHT_DIR}'")
        print("3. 左右目录中是否存在文件名末尾有相同数字编号的图片 (例如: lresult001.png 和 rresult001.png)")
        print(f"4. 图片扩展名是否在支持列表中: {cfg.IMG_EXTS}\n")
        raise RuntimeError("No stereo pairs found. Check LEFT_DIR/RIGHT_DIR.")

    print(f"[信息] 成功找到 {len(pairs)} 对图像。")

    print("[信息] 正在加载DINO模型:", cfg.DINO_MODEL)
    extractor = DINOExtractor(cfg)
    print(f"[信息] DINO模型准备就绪: hidden={extractor.hidden}, patch={extractor.patch}")

    meta = {
        "fx": fx, "baseline_m": B, "n_pairs": len(pairs),
        "cfg": {k: v for k, v in cfg.__dict__.items() if k.isupper()}
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    ps = cfg.PATCH_SIZE
    max_disp_f = max(1, int(math.floor(cfg.MAX_DISP_PX / ps)))

    for name, lpath, rpath in tqdm(pairs, desc="[进度]"):
        L = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
        if L is None or R is None or L.shape != R.shape:
            print(f"[警告] 读取或尺寸不匹配: {name}, 已跳过");
            continue
        H, W = L.shape

        ly, lx = detect_white_dots(L, cfg.DOT_THRESH, cfg.DOT_AREA_MIN, cfg.DOT_AREA_MAX,
                                   cfg.MEDIAN_BLUR, cfg.OPEN_KSIZE)
        if len(lx) == 0:
            print(f"[信息] 左图中未检测到白点: {name}, 已跳过");
            continue

        featL, (Hf, Wf) = extractor.extract_grid(L)
        featR, _ = extractor.extract_grid(R)

        rows, vis_y, vis_x, vis_disp, vis_depth = [], [], [], [], []

        for (yL, xL) in zip(ly, lx):
            y_f, x_f = int(round(yL / ps)), int(round(xL / ps))
            if not (0 <= y_f < Hf and 0 <= x_f < Wf):
                continue

            m = scan_right_along_row(featL, featR, y_f, x_f, max_disp_f,
                                     sim_min=cfg.SIM_MIN, subpix=cfg.SUBPIX)
            if m is None:
                continue
            d_feat, xr_feat, smax = m

            if cfg.CROSS_CHECK:
                if not cross_check(featR, featL, y_f, xr_feat, x_f, max_disp_f,
                                   sim_min=cfg.SIM_MIN, tol=1.0):
                    continue

            disp_px = float(d_feat * ps)
            depth_m = float(disparity_to_depth(disp_px, fx, B))
            rows.append((float(yL), float(xL), disp_px, depth_m, float(smax)))
            vis_y.append(yL);
            vis_x.append(xL);
            vis_disp.append(disp_px);
            vis_depth.append(depth_m)

        stem = os.path.splitext(name)[0]
        if rows:
            save_csv(os.path.join(out_csv, f"{stem}.csv"), rows)
        else:
            print(f"[信息] 未找到有效匹配点: {name}");
            continue

        over = colorize_depth_on_image(L, np.array(vis_y), np.array(vis_x), np.array(vis_depth))
        cv2.imwrite(os.path.join(out_vis, f"{stem}_overlay.png"), over)

        if cfg.MAKE_IDW_PREVIEW and SCIPY_OK and vis_depth:
            depth_map = idw_preview(H, W, np.array(vis_y), np.array(vis_x),
                                    np.array(vis_depth, np.float32),
                                    ds=cfg.IDW_DOWNSAMPLE, k=cfg.IDW_K_NEIGHBORS, p=cfg.IDW_POWER)
            if depth_map is not None:
                valid_depths = depth_map[np.isfinite(depth_map)]
                if valid_depths.size > 0:
                    vmin, vmax = np.percentile(valid_depths, 5), np.percentile(valid_depths, 95)
                    dd_n = np.clip((depth_map - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    dd_n[~np.isfinite(depth_map)] = 0
                    cm = cv2.applyColorMap((dd_n * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(out_vis, f"{stem}_idw_depth.png"), cm)

    print(f"[完成] 所有输出文件已保存至: {out_dir}")


if __name__ == "__main__":
    main()

