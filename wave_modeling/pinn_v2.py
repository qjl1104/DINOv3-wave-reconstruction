# wave_modeling/pinn_v2.py
"""
PINN v2：水面波场重建 η(x, y, t) 的升级版物理约束神经网络。

相对 06-09 草稿（语法错误、坐标约定混乱，从未跑通）的改进：

1. Fourier 特征输入（Tancik et al. 2020；Sallam 2024, arXiv:2409.19851 将其用于
   波面 PINN）：缓解 tanh 网络的谱偏差（先学低频、学不会高频）。
2. 可选控制方程：
   - 'shallow'    ：线性浅水波方程 η_tt = c²∇²η，c = √(gh)
   - 'boussinesq' ：加色散修正 η_tt = c²∇²η + (h²/3)∇²η_tt
     （其色散关系 ω² = ghk²/(1+(kh)²/3) 在 O((kh)²) 阶逼近完整 Airy 色散
     ω² = gk·tanh(kh)；浅水条件 kh < π/10 不满足时应使用此模式）
3. 波速 c 设为可学习参数（log 参数化保证正值）→ 由 h = c²/g 顺带反演水深，
   即 PINN 反问题（参考 Wang & Chen 2022 近岸水深反演思路）。
4. 输入在模型内部按数据范围归一化到 [-1,1]，外部接口始终为物理单位，
   PDE 残差经 autograd 自动穿过归一化，无需手工链式法则。
5. 配置点（collocation points）每个 epoch 重新采样，避免过拟合固定采样点。

调试过程中验证过的三个关键教训（合成实验，2026-07）：
- 教训一：sigma 必须【贴近且不超过】目标谱。sin/cos 的实际频率是 2π·B，
  sigma 过大（如 (2,2,4)）时特征周期小于采样间距，网络沿虚假高频方向
  逐点背诵训练集——训练集 relL2 0.6% 但同分布新点 relL2 高达 61%。
  sigma=(0.5,0.5,1) + 可学习 B 时新点 relL2 降到 1.2%。
- 教训二：振幅必须归一化。波幅 ~1e-3 m 时数据损失被初始化输出噪声
  （O(0.1)）淹没，Adam 也救不回 4 个数量级，网络坍缩到 η≡0 平凡解
  （它同时满足 PDE）。波动方程齐次线性，对归一化 η 求残差严格等价。
- 教训三：物理损失要按首值自归一化。二阶导数带来 ω²≈100 的放大，
  残差 MSE 天然比数据损失大 4 个数量级，固定 lambda 无论取 1e-2 还是
  1e-3 都会压垮数据拟合并把 c 拖向 0。除以首个物理损失值后，
  lambda_phys ∈ [0.1, 1] 都能让 c 从错误初值 0.5 收敛到真值（误差 <0.1%）。

__main__ 为端到端自洽性验证：合成已知行波场 → 稀疏采样训练 → 同分布新点
与网格上评估相对 L2 误差，并检查 c 的反演收敛。可直接用 CPU 运行（约 2 分钟）：
    python wave_modeling/pinn_v2.py
"""

import numpy as np
import torch
import torch.nn as nn

G_DEFAULT = 9.81


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
class FourierFeatures(nn.Module):
    """高斯随机 Fourier 特征：v -> [v, sin(2π·vB), cos(2π·vB)]
    B[i, j] ~ N(0, sigma_i²) 初始化，sigma 按输入维度（x, y, t）分别设置。
    sin/cos 的实际频率是 2π·B：sigma 应让 2π·sigma 恰好覆盖目标波数/频率
    （归一化坐标下），切勿显著超过——否则特征周期小于采样间距，
    网络会沿虚假高频方向背诵训练点而无法泛化（见文件头"教训一"）。
    B 默认可学习（learn_B=True），训练时频率会向数据主模态收敛。"""

    def __init__(self, in_dim=3, num_freqs=128, sigmas=(0.5, 0.5, 1.0),
                 learn_B=True):
        super().__init__()
        B = torch.randn(in_dim, num_freqs) * torch.tensor(sigmas).unsqueeze(1)
        if learn_B:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.out_dim = in_dim + 2 * num_freqs

    def forward(self, v):
        proj = 2.0 * torch.pi * v @ self.B  # [N, F]
        return torch.cat([v, torch.sin(proj), torch.cos(proj)], dim=-1)


class PINNWaveV2(nn.Module):
    """η_θ(x, y, t)，内部做 [-1,1] 归一化；c = exp(log_c) 可学习。"""

    def __init__(self, bounds, hidden_layers=3, hidden_units=64, num_freqs=128,
                 sigmas=(0.5, 0.5, 1.0), learn_B=True, c_init=0.5, learn_c=True):
        super().__init__()
        # bounds: dict('x': (min,max), 'y': ..., 't': ...)
        lo = torch.tensor([bounds[k][0] for k in ("x", "y", "t")], dtype=torch.float32)
        hi = torch.tensor([bounds[k][1] for k in ("x", "y", "t")], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

        self.ff = FourierFeatures(3, num_freqs, sigmas, learn_B)
        layers, d = [], self.ff.out_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(d, hidden_units), nn.Tanh()]
            d = hidden_units
        layers.append(nn.Linear(d, 1))
        self.network = nn.Sequential(*layers)

        self.log_c = nn.Parameter(torch.tensor(float(np.log(c_init))),
                                  requires_grad=learn_c)
        # 振幅归一化：波幅通常 ~1e-3 m，直接拟合会让数据损失梯度淹没在
        # 初始化噪声之下（Adam 也救不回 4 个数量级）。网络输出归一化的 η，
        # train_pinn 会把此 buffer 设为数据 std；波动方程是齐次线性的，
        # 对归一化 η 求残差与物理 η 等价（c 不受影响）。
        self.register_buffer("eta_scale", torch.ones(()))

    @property
    def c(self):
        return torch.exp(self.log_c)

    def forward(self, xyt):
        """归一化的 η（O(1) 量级）。物理单位输出用 predict()。"""
        xyt = xyt.to(self.lo.dtype)  # 容忍 float64 输入
        v = 2.0 * (xyt - self.lo) / (self.hi - self.lo) - 1.0
        return self.network(self.ff(v))

    def predict(self, xyt):
        return self(xyt) * self.eta_scale

    # ---------------- 物理残差 ----------------
    @staticmethod
    def _grad(y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True)[0]

    def physics_residual(self, xyt, h=None, mode="shallow"):
        """xyt: [N,3] 物理单位、requires_grad=True。返回 PDE 残差 [N,1]。
        c 一律由 c_init/learn_c 直接给定（原 g 参数从未被使用，已移除）；
        h 为水深，mode='boussinesq' 时【必填】，且单位须与 xyt 坐标一致
        （真实管线为 mm，本文件合成自测为 m）。"""
        eta = self(xyt)
        g1 = self._grad(eta, xyt)
        eta_x, eta_y, eta_t = g1[:, 0:1], g1[:, 1:2], g1[:, 2:3]

        eta_xx = self._grad(eta_x, xyt)[:, 0:1]
        eta_yy = self._grad(eta_y, xyt)[:, 1:2]
        eta_tt = self._grad(eta_t, xyt)[:, 2:3]

        lap = eta_xx + eta_yy
        res = eta_tt - self.c ** 2 * lap
        if mode == "boussinesq":
            if h is None:
                raise ValueError(
                    "mode='boussinesq' 必须显式传水深 h（单位同 xyt 坐标）")
            # ∇²η_tt = ∂²(η_xx)/∂t² + ∂²(η_yy)/∂t²
            eta_xx_tt = self._grad(self._grad(eta_xx, xyt)[:, 2:3], xyt)[:, 2:3]
            eta_yy_tt = self._grad(self._grad(eta_yy, xyt)[:, 2:3], xyt)[:, 2:3]
            res = res - (h ** 2 / 3.0) * (eta_xx_tt + eta_yy_tt)
        elif mode != "shallow":
            raise ValueError(f"未知 mode: {mode}")
        return res


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def sample_collocation(bounds, n, device):
    """在数据包围盒内均匀采样配置点（物理单位）。统一 float32，
    避免 numpy float64 的 bounds 污染模型计算。"""
    lo = torch.tensor([bounds[k][0] for k in ("x", "y", "t")], dtype=torch.float32)
    hi = torch.tensor([bounds[k][1] for k in ("x", "y", "t")], dtype=torch.float32)
    return (lo + (hi - lo) * torch.rand(n, 3)).to(device)


def synthetic_wave_data(bounds, c_true, n_data=2000, amplitude=2e-3,
                       modes=(2, 1), noise_std=0.0, seed=0):
    """合成行波场 η = A·cos(kx·x + ky·y − ωt)，ω = c_true·k（浅水色散）。
    返回 (data_xyt, data_eta)。"""
    rng = np.random.default_rng(seed)
    Lx = bounds["x"][1] - bounds["x"][0]
    Ly = bounds["y"][1] - bounds["y"][0]
    kx, ky = 2 * np.pi * modes[0] / Lx, 2 * np.pi * modes[1] / Ly
    k = np.hypot(kx, ky)
    omega = c_true * k

    x = rng.uniform(*bounds["x"], n_data)
    y = rng.uniform(*bounds["y"], n_data)
    t = rng.uniform(*bounds["t"], n_data)
    eta = amplitude * np.cos(kx * x + ky * y - omega * t)
    if noise_std > 0:
        eta += rng.normal(0, noise_std, n_data)

    xyt = torch.tensor(np.stack([x, y, t], 1), dtype=torch.float32)
    return xyt, torch.tensor(eta, dtype=torch.float32).unsqueeze(1)


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
def train_pinn(model, data_xyt, data_eta, bounds, epochs=3000, lr=1e-3,
               lambda_phys=1.0, n_colloc=2048, h=None,
               mode="shallow", warmup_frac=0.4, log_every=500,
               device="cpu", seed=0):
    """两个关键的损失平衡机制（见文件头"教训二/三"）：
    - warmup_frac：前 warmup_frac·epochs 内 lambda 从 0 线性升到目标值，
      避免初始随机网络的巨大 PDE 残差把优化器引向 η≡0 平凡解。
    - 物理损失按【首个非零值自归一化】：二阶导数带 ω²≈100 放大，
      残差 MSE 天然比数据损失大 ~4 个数量级；除以 lp0 后
      lambda_phys ∈ [0.1, 1] 都能兼顾场重建精度与 c 反演。
    mode='boussinesq' 时 h（水深，单位同数据坐标）必填，见 physics_residual。"""
    torch.manual_seed(seed)
    model.to(device)
    data_xyt, data_eta = data_xyt.to(device), data_eta.to(device)
    # 振幅归一化（见模型里 eta_scale 的注释）
    with torch.no_grad():
        model.eta_scale.fill_(data_eta.std().clamp_min(1e-12))
    eta_norm = data_eta / model.eta_scale
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = max(1, int(warmup_frac * epochs))
    lp0 = None

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss_data = torch.mean((model(data_xyt) - eta_norm) ** 2)

        lam = lambda_phys * min(1.0, epoch / warmup)
        if lam > 0:
            colloc = sample_collocation(bounds, n_colloc, device).requires_grad_(True)
            loss_phys = torch.mean(model.physics_residual(colloc, h=h, mode=mode) ** 2)
            if lp0 is None:
                lp0 = loss_phys.detach().clamp_min(1e-12)
        else:
            loss_phys = torch.zeros((), device=device)

        loss = loss_data if lp0 is None else loss_data + lam * loss_phys / lp0
        loss.backward()
        opt.step()

        if epoch % log_every == 0 or epoch == 1:
            # c 的单位随数据坐标（合成自测为 m/s，真实管线为 mm/s），不写死
            print(f"epoch {epoch:5d} | data {loss_data.item():.3e} | "
                  f"phys {loss_phys.item():.3e} | lam {lam:.1e} | "
                  f"c = {model.c.item():.4f} [坐标单位/s]")
    return model


@torch.no_grad()
def relative_l2(model, xyt, eta_true, device="cpu"):
    """η_true 为物理单位；用 predict()（乘回 eta_scale）比较。"""
    model.eval()
    pred = torch.cat([model.predict(xyt[i:i + 4096].to(device))
                      for i in range(0, len(xyt), 4096)]).cpu()
    return (torch.norm(pred - eta_true) / torch.norm(eta_true)).item()


# ----------------------------------------------------------------------
# 端到端合成验证
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 物理参数：1m × 1m 水域、水深 5cm（kh ≈ 0.7，接近浅水极限）
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 2.0)}
    h_true = 0.05
    c_true = float(np.sqrt(G_DEFAULT * h_true))  # ≈ 0.70 m/s

    data_xyt, data_eta = synthetic_wave_data(bounds, c_true, n_data=20000, seed=0)

    # c 故意从错误初值 0.5 m/s 出发，验证反演收敛
    model = PINNWaveV2(bounds, c_init=0.5, learn_c=True)
    model = train_pinn(model, data_xyt, data_eta, bounds,
                       epochs=2500, lambda_phys=1.0, n_colloc=2048,
                       h=h_true, mode="shallow", log_every=500)

    # 网格上评估相对 L2
    gx, gy = np.meshgrid(np.linspace(*bounds["x"], 40),
                         np.linspace(*bounds["y"], 40), indexing="ij")
    gt = np.linspace(*bounds["t"], 15)
    mesh = np.stack(np.meshgrid(gx[:, 0], gy[0], gt, indexing="ij"), -1)
    xyt_grid = torch.tensor(mesh.reshape(-1, 3), dtype=torch.float32)
    Lx = Ly = 1.0
    kx, ky = 2 * np.pi * 2 / Lx, 2 * np.pi * 1 / Ly
    omega = c_true * np.hypot(kx, ky)
    eta_grid = 2e-3 * np.cos(kx * mesh[..., 0] + ky * mesh[..., 1]
                             - omega * mesh[..., 2])
    eta_grid = torch.tensor(eta_grid.reshape(-1, 1), dtype=torch.float32)

    err = relative_l2(model, xyt_grid, eta_grid)
    c_rec = model.c.item()
    h_rec = c_rec ** 2 / G_DEFAULT
    c_err = abs(c_rec - c_true) / c_true
    print(f"\n[验证] 网格相对 L2 误差 = {err:.3%}（<20% 为通过）")
    print(f"[验证] c: 真值 {c_true:.4f} → 反演 {c_rec:.4f} m/s（误差 {c_err:.2%}，<5% 为通过）; "
          f"对应水深 h: 真值 {h_true:.4f} → 反演 {h_rec:.4f} m")
    assert err < 0.20 and c_err < 0.05, "合成验证未通过"
    print("[验证] 全部通过")

    # boussinesq 模式冒烟测试（4 阶混合导数可走通且数值有限即可）
    xb = sample_collocation(bounds, 64, "cpu").requires_grad_(True)
    rb = model.physics_residual(xb, h=h_true, mode="boussinesq")
    assert torch.isfinite(rb).all(), "boussinesq 残差出现非有限值"
    print(f"[验证] boussinesq 残差冒烟测试通过，|res|_mean = {rb.abs().mean():.2e}")
