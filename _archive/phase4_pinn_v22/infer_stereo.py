# -*- coding: utf-8 -*-
"""
使用训练好的 best_model_self_supervised.pth 对单对图像推理，输出视差（u16）和伪彩图；
可选：提供 fx 与基线（或 npz）以导出深度（米）。
"""

import os, argparse, cv2, numpy as np, torch
from dinov3_wave_stereo import Config, DINOv3StereoModel  # 确保同目录下已保存训练脚本

def imread_gray3(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return np.stack([img]*3, axis=-1)

def save_color_disparity(disp, out_png):
    disp_norm = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
    disp_u8 = (disp_norm * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)
    cv2.imwrite(out_png, disp_color)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--left', required=True, help='左图路径')
    ap.add_argument('--right', required=True, help='右图路径')
    ap.add_argument('--ckpt', required=True, help='训练得到的 best_model_self_supervised.pth')
    ap.add_argument('--out_dir', default='./infer_out')
    ap.add_argument('--calib_npz', default='', help='可选：包含 fx/baseline 的 npz')
    ap.add_argument('--fx', type=float, default=0.0, help='焦距（像素）')
    ap.add_argument('--baseline', type=float, default=0.0, help='基线（米）')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 构建与训练一致的 Config（重要：保持 ViT-B/16 + Tiling）
    cfg = Config()
    cfg.PRESERVE_ORIGINAL_RES = True
    cfg.USE_TILING = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINOv3StereoModel(cfg).to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"✓ loaded checkpoint: {args.ckpt}")

    # 读图 → 张量
    L = imread_gray3(args.left)
    R = imread_gray3(args.right)
    Lt = torch.from_numpy(L.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    Rt = torch.from_numpy(R.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    Lt, Rt = Lt.to(device), Rt.to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        out = model(Lt, Rt)
        disp = out['disparity'][0,0].float().cpu().numpy()  # H×W, 像素单位

    # 保存视差（16-bit 原值）与伪彩
    disp16 = np.clip(disp, 0, 65535).astype(np.uint16)
    out_disp_raw = os.path.join(args.out_dir, 'disparity_raw_u16.png')
    out_disp_col = os.path.join(args.out_dir, 'disparity_color.png')
    cv2.imwrite(out_disp_raw, disp16)
    save_color_disparity(disp, out_disp_col)
    print(f"✓ saved: {out_disp_raw}\n✓ saved: {out_disp_col}")

    # 可选：用 fx/基线算深度（米）
    fx = args.fx; B = args.baseline
    if args.calib_npz and (fx==0 or B==0):
        try:
            calib = np.load(args.calib_npz)
            # 这里尝试几个常见键名；若你的 npz 键名不同，请改成对应名字
            fx = float(calib.get('fx', calib.get('K_left', np.array([[fx]]) )[0][0]))
            if 'baseline' in calib:
                B = float(calib['baseline'])
            elif 'T' in calib:  # 若保存的是外参平移向量（单位米）
                T = calib['T'].reshape(-1)
                B = abs(float(T[0]))
        except Exception as e:
            print(f"[warn] 读取 calib_npz 失败: {e}")

    if fx > 0 and B > 0:
        depth = fx * B / np.maximum(disp, 1e-6)
        np.save(os.path.join(args.out_dir, 'depth_m.npy'), depth)
        print("✓ saved: depth_m.npy (米)")
    else:
        print("提示：若要导出深度，请提供 --fx 与 --baseline，或在 calib_npz 里包含同名键。")

if __name__ == "__main__":
    main()
