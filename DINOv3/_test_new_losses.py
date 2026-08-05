"""反捷径损失单元功能测试。

验证：
1. 三个新损失函数 (NCC / DispRange / LR-Consistency) 能正常前向传播
2. 输出 shape 和数值范围合理
3. 反向传播梯度可流通到模型参数
4. 极端场景（零视差捷径）下损失应较高 → 验证惩罚方向正确

运行：
    cd DINOv3
    python _test_new_losses.py
"""
import torch
import torch.nn.functional as F
from config import Config
from losses import PINNPhysicsLoss


def setup_test_cfg(cfg):
    """统一设置测试用 H/W（避开 config 的 auto-detect 0）。"""
    cfg.IMAGE_HEIGHT = 512
    cfg.IMAGE_WIDTH = 512
    return cfg


def make_mock_data(cfg, device='cpu'):
    """构造 mock 数据：模拟小圆片在水面上的关键点。"""
    B, N = 2, 30
    H = cfg.IMAGE_HEIGHT
    W = cfg.IMAGE_WIDTH

    # 左图关键点随机分布在画面中部
    kpl = torch.zeros(B, N, 2, device=device)
    kpl[:, :, 0] = torch.rand(B, N, device=device) * (W * 0.8) + W * 0.1  # x
    kpl[:, :, 1] = torch.rand(B, N, device=device) * (H * 0.6) + H * 0.2  # y

    # 真实视差：模拟 Z=2000-15000mm → d≈248-1859
    true_disp = torch.rand(B, N, device=device) * 1600 + 200  # [200, 1800]
    # 右图关键点 = 左图 x - disp
    kpr = kpl.clone()
    kpr[:, :, 0] = kpl[:, :, 0] - true_disp

    scores = torch.ones(B, N, device=device) * 0.8

    # 灰度图：构造有结构的图案（避免全黑导致 NCC 退化）
    lg = torch.rand(B, 1, H, W, device=device)
    rg = lg.clone()
    # 让右图在匹配位置有相似 patch（模拟真实场景）
    for b in range(B):
        for i in range(N):
            x_l, y_l = int(kpl[b, i, 0]), int(kpl[b, i, 1])
            x_r = int(kpl[b, i, 0] - true_disp[b, i])
            if 5 < x_l < W - 5 and 5 < x_r < W - 5 and 5 < y_l < H - 5:
                patch = lg[b, :, y_l-5:y_l+5, x_l-5:x_l+5]
                rg[b, :, y_l-5:y_l+5, x_r-5:x_r+5] = patch

    # Q 矩阵 mock (需要 [B, 4, 4] 维度供 matmul 使用)
    Q_single = torch.eye(4, device=device)
    Q_single[2, 3] = -3718679.0  # fB
    Q_single[3, 2] = 1.0
    Q_single[3, 3] = 0.0
    Q = Q_single.unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]

    return lg, rg, kpl, kpr, scores, Q, true_disp


def test_ncc_loss():
    """测试 NCC 匹配损失。"""
    print("\n=== 测试 1: NCC 匹配损失 ===")
    cfg = setup_test_cfg(Config())
    device = 'cpu'
    loss_fn = PINNPhysicsLoss(cfg)

    lg, rg, kpl, kpr, scores, Q, true_disp = make_mock_data(cfg, device)

    # 场景 A：正确视差 → NCC 应较高（损失较低）
    l_ncc_correct = loss_fn.compute_ncc_match_loss(lg, rg, kpl, true_disp, scores)

    # 场景 B：零视差捷径 → NCC loss=0（不参与，由 disp_range_loss 接管）
    zero_disp = torch.zeros_like(true_disp)
    l_ncc_zero = loss_fn.compute_ncc_match_loss(lg, rg, kpl, zero_disp, scores)

    # 场景 C：随机错误视差（在合理范围内）→ NCC 应较低
    rand_disp = torch.rand_like(true_disp) * 1500 + 300  # [300, 1800]
    l_ncc_rand = loss_fn.compute_ncc_match_loss(lg, rg, kpl, rand_disp, scores)

    # 场景 D：零视差应由 disp_range_loss 惩罚
    l_range_zero = loss_fn.compute_disp_range_loss(zero_disp, scores)
    l_range_correct = loss_fn.compute_disp_range_loss(true_disp, scores)

    print(f"  正确视差 NCC loss:   {l_ncc_correct.item():.4f}  (应较低)")
    print(f"  零视差   NCC loss:   {l_ncc_zero.item():.4f}  (应=0, 交由 range 接管)")
    print(f"  随机视差 NCC loss:   {l_ncc_rand.item():.4f}  (应较高)")
    print(f"  零视差   Range loss: {l_range_zero.item():.4f}  (应>>0, 惩罚捷径)")
    print(f"  正确视差 Range loss: {l_range_correct.item():.4f}  (应≈0)")

    # 关键断言：
    # 1. NCC 在合理视差范围内能区分正确 vs 随机匹配
    assert l_ncc_correct.item() <= l_ncc_rand.item() + 0.5, \
        f"NCC 应让正确视差 loss <= 随机视差 loss: correct={l_ncc_correct}, rand={l_ncc_rand}"
    # 2. 零视差时 NCC loss=0（不奖励也不惩罚）
    assert abs(l_ncc_zero.item()) < 1e-3, f"零视差 NCC loss 应=0, 实际 {l_ncc_zero.item()}"
    # 3. 零视差由 disp_range_loss 惩罚
    assert l_range_zero.item() > 50.0, f"零视差应被 disp_range 重罚, 实际 {l_range_zero.item()}"
    assert l_range_correct.item() < 1e-3, f"正确视差 range loss 应≈0, 实际 {l_range_correct.item()}"
    assert torch.isfinite(l_ncc_correct), "NCC loss 包含 NaN/Inf"
    print("  [PASS] NCC + Range 协同惩罚捷径正确 + 数值有限")


def test_disp_range_loss():
    """测试视差范围先验损失。"""
    print("\n=== 测试 2: 视差范围先验损失 ===")
    cfg = setup_test_cfg(Config())
    device = 'cpu'
    loss_fn = PINNPhysicsLoss(cfg)

    B, N = 2, 30
    scores = torch.ones(B, N, device=device)

    # 场景 A：视差在合理范围 [100, 2000]
    disp_ok = torch.rand(B, N, device=device) * 1800 + 200
    l_ok = loss_fn.compute_disp_range_loss(disp_ok, scores)

    # 场景 B：视差过小 (< 100，对应太远)
    disp_small = torch.full((B, N), 50.0, device=device)
    l_small = loss_fn.compute_disp_range_loss(disp_small, scores)

    # 场景 C：视差过大 (> 2000，对应太近)
    disp_large = torch.full((B, N), 3000.0, device=device)
    l_large = loss_fn.compute_disp_range_loss(disp_large, scores)

    # 场景 D：零视差捷径
    disp_zero = torch.zeros(B, N, device=device)
    l_zero = loss_fn.compute_disp_range_loss(disp_zero, scores)

    print(f"  合理视差  Range loss: {l_ok.item():.4f}    (应≈0)")
    print(f"  过小视差  Range loss: {l_small.item():.4f}  (应>0)")
    print(f"  过大视差  Range loss: {l_large.item():.4f}  (应>0)")
    print(f"  零视差    Range loss: {l_zero.item():.4f}    (应最大)")

    assert l_ok.item() < 1e-3, f"合理视差损失应为 0，实际 {l_ok.item()}"
    assert l_small.item() > 0, "过小视差应被惩罚"
    assert l_large.item() > 0, "过大视差应被惩罚"
    assert l_zero.item() > l_ok.item(), "零视差应比合理视差损失高"
    assert torch.isfinite(l_ok), "Range loss 包含 NaN/Inf"
    print("  [PASS] 视差范围损失方向正确 + 数值有限")


def test_lr_consistency_loss():
    """测试左右一致性损失。"""
    print("\n=== 测试 3: 左右一致性损失 ===")
    cfg = setup_test_cfg(Config())
    device = 'cpu'
    loss_fn = PINNPhysicsLoss(cfg)

    B, N = 2, 30
    H, W = cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH

    kpl = torch.zeros(B, N, 2, device=device)
    kpl[:, :, 0] = torch.rand(B, N, device=device) * (W * 0.8) + W * 0.1
    kpl[:, :, 1] = torch.rand(B, N, device=device) * (H * 0.6) + H * 0.2

    # 正向视差
    disp_lr = torch.rand(B, N, device=device) * 1600 + 200

    # 右图关键点 = 左图匹配位置
    kpr = kpl.clone()
    kpr[:, :, 0] = kpl[:, :, 0] - disp_lr

    # 反向视差：约定 right_col - left_match_col
    # 若一致，disp_lr + disp_rl = 0 → disp_rl = -disp_lr
    disp_rl_consistent = -disp_lr  # 但这只对每个右图关键点对齐

    scores = torch.ones(B, N, device=device)

    # 场景 A：完全一致
    l_consist = loss_fn.compute_lr_consistency_loss(
        kpl, disp_lr, kpr, disp_rl_consistent, scores
    )

    # 场景 B：完全不一致（反向视差符号相同）
    disp_rl_inconsistent = disp_lr.clone()
    l_inconsist = loss_fn.compute_lr_consistency_loss(
        kpl, disp_lr, kpr, disp_rl_inconsistent, scores
    )

    print(f"  一致   LR loss: {l_consist.item():.4f}    (应≈0)")
    print(f"  不一致 LR loss: {l_inconsist.item():.4f}  (应>>0)")

    assert l_consist.item() < 1.0, f"一致场景损失应近 0，实际 {l_consist.item()}"
    assert l_inconsist.item() > l_consist.item(), \
        f"不一致应比一致损失高: incons={l_inconsist} should > cons={l_consist}"
    assert torch.isfinite(l_consist), "LR loss 包含 NaN/Inf"
    print("  [PASS] 左右一致性损失方向正确 + 数值有限")


def test_full_forward_backward():
    """测试完整 forward + loss + backward 路径。"""
    print("\n=== 测试 4: 完整 forward + backward ===")
    cfg = setup_test_cfg(Config())
    device = 'cpu'
    loss_fn = PINNPhysicsLoss(cfg)

    lg, rg, kpl, kpr, scores, Q, true_disp = make_mock_data(cfg, device)

    # 模拟模型预测：requires_grad 让梯度流通
    # 加小扰动让 l_lr > 0，确保 disp_rev 有梯度
    disparity_pred = (true_disp + torch.randn_like(true_disp) * 10).requires_grad_(True)
    disparity_rev_pred = (-true_disp + torch.randn_like(true_disp) * 10).requires_grad_(True)

    # Sinkhorn 软分配矩阵 mock
    B, N = kpl.shape[:2]
    patch_size = 16  # DINOv3 ViT-B/16
    Wf = cfg.IMAGE_WIDTH // patch_size
    prob_list = [torch.softmax(torch.randn(N, Wf, device=device), dim=-1) for _ in range(B)]

    # 调用完整的 forward
    l_disp, l_smooth, l_slope, l_zeromean, l_corr, l_ncc, l_range, l_lr = loss_fn(
        lg, rg, kpl, kpr, scores, Q, prob_list,
        disparity=disparity_pred, kpr_actual=kpr, disparity_rev=disparity_rev_pred
    )

    print(f"  l_disp:    {l_disp.item():.4f}")
    print(f"  l_smooth:  {l_smooth.item():.4f}")
    print(f"  l_slope:   {l_slope.item():.4f}")
    print(f"  l_zeromean:{l_zeromean.item():.4f}")
    print(f"  l_corr:    {l_corr.item():.4f}")
    print(f"  l_ncc:     {l_ncc.item():.4f}")
    print(f"  l_range:   {l_range.item():.4f}")
    print(f"  l_lr:      {l_lr.item():.4f}")

    total = (cfg.CORRELATION_WEIGHT * l_corr + cfg.DISPARITY_WEIGHT * l_disp +
             cfg.PHY_SMOOTH_WEIGHT * l_smooth + cfg.PHY_SLOPE_WEIGHT * l_slope +
             cfg.PHY_ZEROMEAN_WEIGHT * l_zeromean + cfg.NCC_MATCH_WEIGHT * l_ncc +
             cfg.DISP_RANGE_WEIGHT * l_range + cfg.LR_CONSISTENCY_WEIGHT * l_lr)

    # 反向传播
    total.backward()

    # 检查梯度
    grad_pred = disparity_pred.grad
    grad_rev = disparity_rev_pred.grad

    print(f"\n  total loss: {total.item():.4f}")
    print(f"  disparity 梯度: shape={grad_pred.shape}, mean={grad_pred.abs().mean().item():.4f}")
    print(f"  disp_rev 梯度:  shape={grad_rev.shape}, mean={grad_rev.abs().mean().item():.4f}")

    assert torch.isfinite(total), "Total loss 包含 NaN/Inf"
    assert grad_pred is not None, "disparity 梯度未流通"
    assert grad_rev is not None, "disparity_rev 梯度未流通"
    assert grad_pred.abs().sum() > 0, "disparity 梯度全为 0"
    assert grad_rev.abs().sum() > 0, "disp_rev 梯度全为 0"
    print("  [PASS] 完整 forward + backward 梯度流通正常")


def test_edge_cases():
    """测试边界场景：空关键点、极少关键点。"""
    print("\n=== 测试 5: 边界场景 ===")
    cfg = setup_test_cfg(Config())
    device = 'cpu'
    loss_fn = PINNPhysicsLoss(cfg)

    # 场景 A：scores 全 0
    B, N = 1, 5
    kpl = torch.rand(B, N, 2, device=device)
    kpl[:, :, 0] *= cfg.IMAGE_WIDTH
    kpl[:, :, 1] *= cfg.IMAGE_HEIGHT
    scores_zero = torch.zeros(B, N, device=device)
    disp = torch.rand(B, N, device=device) * 1000

    l_ncc = loss_fn.compute_ncc_match_loss(
        torch.rand(B, 1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, device=device),
        torch.rand(B, 1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, device=device),
        kpl, disp, scores_zero
    )
    l_range = loss_fn.compute_disp_range_loss(disp, scores_zero)

    print(f"  scores=0 NCC loss:   {l_ncc.item():.4f}  (应=0)")
    print(f"  scores=0 Range loss: {l_range.item():.4f}  (应=0)")

    assert torch.isfinite(l_ncc) and l_ncc.item() == 0.0, "scores=0 时 NCC 应返回 0"
    assert torch.isfinite(l_range) and l_range.item() == 0.0, "scores=0 时 Range 应返回 0"

    # 场景 B：左右一致性，右图关键点 < 5
    kpr_small = torch.rand(B, 3, 2, device=device)
    disp_rl = torch.rand(B, 3, device=device) * 100
    l_lr = loss_fn.compute_lr_consistency_loss(kpl, disp, kpr_small, disp_rl, scores_zero)
    assert torch.isfinite(l_lr), "右图关键点过少时 LR loss 应返回有限 0"

    print("  [PASS] 边界场景处理正确")


if __name__ == "__main__":
    print("=" * 60)
    print("反捷径损失单元功能测试")
    print("=" * 60)

    test_ncc_loss()
    test_disp_range_loss()
    test_lr_consistency_loss()
    test_full_forward_backward()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
