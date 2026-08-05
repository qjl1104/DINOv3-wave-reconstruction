"""诊断 NCC loss 不下降的根因。

加载 best_model，在验证集上：
1. 统计预测视差分布
2. 检查 patch 内容（均值/方差/有效像素比）
3. 对比正确匹配 vs 错误匹配的 NCC 值
4. 检查 grid_sample 梯度是否流通

运行：
    cd DINOv3
    python _diagnose_ncc.py
"""
import torch
import numpy as np
from config import Config
from models import CorrMatchingStereoModel
from dataset import RectifiedWaveStereoDataset
from losses import PINNPhysicsLoss
from utils import pad_to_patch_size


def main():
    cfg = Config()
    device = 'cuda'

    print("=== 加载模型和数据 ===")
    model = CorrMatchingStereoModel(cfg).to(device)
    ckpt_path = "training_runs/20260716-152604/checkpoints/best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"已加载: {ckpt_path} (epoch={ckpt.get('epoch', '?')})")

    dataset = RectifiedWaveStereoDataset(cfg, is_validation=True)
    loss_fn = PINNPhysicsLoss(cfg)

    # 取前 3 个样本诊断
    for idx in range(3):
        print(f"\n{'='*60}")
        print(f"样本 {idx}")
        print(f"{'='*60}")

        sample = dataset[idx]
        lg = sample['left_gray'].unsqueeze(0).to(device)
        rg = sample['right_gray'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        Q = sample['Q'].unsqueeze(0).to(device)

        is_cached = sample.get('cached', False)
        if is_cached:
            cached_data = {
                'keypoints_left': sample['keypoints_left'].unsqueeze(0).to(device),
                'scores_left': sample['scores_left'].unsqueeze(0).to(device),
                'keypoints_right': sample['keypoints_right'].unsqueeze(0).to(device),
                'scores_right': sample['scores_right'].unsqueeze(0).to(device),
                'feat_left': sample['feat_left'].unsqueeze(0).to(device).float(),
                'feat_right': sample['feat_right'].unsqueeze(0).to(device).float(),
            }
            lrgb = rrgb = None
            if rg is None:
                rg = torch.zeros_like(lg)
        else:
            lrgb = sample['left_rgb'].unsqueeze(0).to(device)
            rrgb = sample['right_rgb'].unsqueeze(0).to(device)
            cached_data = None
            patch_size = model.ext.patch
            lg, rg, lrgb, rrgb, mask = pad_to_patch_size(lg, rg, lrgb, rrgb, mask, patch_size=patch_size)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            with torch.no_grad():
                out = model(lg, rg, lrgb, rrgb, mask, cached_data=cached_data)

        disparity = out['disparity']  # [1, N]
        scores = out['scores_left']   # [1, N]
        kpl = out['keypoints_left']   # [1, N, 2]

        # 1. 视差统计
        print("\n--- 1. 预测视差统计 ---")
        valid = scores > 0.1
        disp_valid = disparity[valid]
        print(f"  关键点总数: {disparity.shape[1]}")
        print(f"  有效关键点 (score>0.1): {valid.sum().item()}")
        print(f"  视差范围: [{disp_valid.min().item():.1f}, {disp_valid.max().item():.1f}]")
        print(f"  视差均值: {disp_valid.mean().item():.1f}")
        print(f"  视差中位: {disp_valid.median().item():.1f}")
        print(f"  视差 < 100 (太远): {(disp_valid < 100).sum().item()} ({(disp_valid < 100).float().mean().item()*100:.1f}%)")
        print(f"  视差 > 2000 (太近): {(disp_valid > 2000).sum().item()}")
        print(f"  视差在 [248, 1859] (物理合理): {((disp_valid >= 248) & (disp_valid <= 1859)).sum().item()} "
              f"({((disp_valid >= 248) & (disp_valid <= 1859)).float().mean().item()*100:.1f}%)")

        # 2. Patch 内容分析
        print("\n--- 2. Patch 内容分析 ---")
        ps = cfg.PATCH_SIZE_PHOTOMETRIC  # 11
        # 注意：cached 模式下 lg 未 padding，但 sample_patches 只用 kpl 坐标采样
        patches_l = loss_fn.sample_patches(lg.float(), kpl.float(), ps)  # [1, C, N, ps, ps]
        print(f"  Patch 大小: {ps}x{ps}")
        print(f"  左图 patch 形状: {patches_l.shape}")

        # 检查 patch 的亮度分布
        patch_means = patches_l.mean(dim=[1, 3, 4])  # [1, N]
        patch_stds = patches_l.std(dim=[1, 3, 4])    # [1, N]
        # 有效像素（非零）比例
        nonzero_ratio = (patches_l[:, 0] > 5.0).float().mean(dim=[1, 2])  # [1, N] per patch

        print(f"  Patch 像素均值: mean={patch_means.mean().item():.1f}, std={patch_means.std().item():.1f}")
        print(f"  Patch 像素标准差: mean={patch_stds.mean().item():.1f}")
        print(f"  Patch 非零像素比: mean={nonzero_ratio.mean().item():.3f}")
        print(f"  Patch 全黑 (<5) 占比: {(patch_means < 5).float().mean().item()*100:.1f}%")

        # 3. NCC 对比：正确匹配 vs 零视差 vs 随机
        print("\n--- 3. NCC 值对比 ---")

        # 用模型预测的视差采样右图 patch
        kpr_pred = kpl.float().clone()
        kpr_pred[..., 0] = kpl[..., 0] - disparity.float()
        patches_r_pred = loss_fn.sample_patches(rg.float(), kpr_pred, ps)

        # 零视差（同位置）
        patches_r_zero = patches_l.clone()

        # 随机偏移视差
        rand_disp = torch.rand_like(disparity.float()) * 1500 + 200
        kpr_rand = kpl.float().clone()
        kpr_rand[..., 0] = kpl[..., 0] - rand_disp
        patches_r_rand = loss_fn.sample_patches(rg.float(), kpr_rand, ps)

        def compute_ncc(pl, pr):
            B, C, N = pl.shape[:3]
            pl_f = pl.reshape(B, C, N, -1)
            pr_f = pr.reshape(B, C, N, -1)
            pl_n = torch.nn.functional.normalize(pl_f - pl_f.mean(-1, keepdim=True), dim=-1)
            pr_n = torch.nn.functional.normalize(pr_f - pr_f.mean(-1, keepdim=True), dim=-1)
            return (pl_n * pr_n).sum(-1).mean(1)  # [B, N]

        ncc_pred = compute_ncc(patches_l, patches_r_pred)
        ncc_zero = compute_ncc(patches_l, patches_r_zero)
        ncc_rand = compute_ncc(patches_l, patches_r_rand)

        print(f"  NCC (模型预测视差): mean={ncc_pred.mean().item():.4f}, std={ncc_pred.std().item():.4f}")
        print(f"  NCC (零视差/同位置): mean={ncc_zero.mean().item():.4f}, std={ncc_zero.std().item():.4f}")
        print(f"  NCC (随机视差):      mean={ncc_rand.mean().item():.4f}, std={ncc_rand.std().item():.4f}")

        # 关键判断：三者是否有区别？
        print(f"\n  → 预测 vs 零视差 差异: {(ncc_pred.mean() - ncc_zero.mean()).item():.4f}")
        print(f"  → 预测 vs 随机 差异:   {(ncc_pred.mean() - ncc_rand.mean()).item():.4f}")
        if abs(ncc_pred.mean() - ncc_rand.mean()) < 0.01:
            print("  ⚠ 预测视差和随机视差的 NCC 几乎相同 → patch 匹配无信号！")
        if ncc_zero.mean() > 0.99:
            print("  ⚠ 零视差 NCC≈1 → patch 内容几乎相同（全黑或结构重复）")

        # 4. 梯度流通检查
        print("\n--- 4. 梯度流通检查 ---")
        disparity_grad_test = disparity.clone().float().requires_grad_(True)
        kpr_test = kpl.float().clone()
        kpr_test[..., 0] = kpl[..., 0] - disparity_grad_test
        patches_r_test = loss_fn.sample_patches(rg.float(), kpr_test, ps)

        pl = patches_l.reshape(1, 1, -1, ps*ps)
        pr = patches_r_test.reshape(1, 1, -1, ps*ps)
        pl_n = torch.nn.functional.normalize(pl - pl.mean(-1, keepdim=True), dim=-1)
        pr_n = torch.nn.functional.normalize(pr - pr.mean(-1, keepdim=True), dim=-1)
        ncc_test = (pl_n * pr_n).sum(-1).mean()
        ncc_test.backward()

        grad = disparity_grad_test.grad
        print(f"  disparity 梯度: mean={grad.abs().mean().item():.6f}, max={grad.abs().max().item():.6f}")
        print(f"  梯度为零的比例: {(grad.abs() < 1e-8).float().mean().item()*100:.1f}%")
        if grad.abs().mean() < 1e-4:
            print("  ⚠ 梯度极小！NCC loss 无法有效反传到 disparity → 这就是 NCC 不下降的原因")


if __name__ == "__main__":
    main()
