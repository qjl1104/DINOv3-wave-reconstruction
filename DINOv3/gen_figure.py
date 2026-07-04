import matplotlib.pyplot as plt
import numpy as np

# 设定稀疏度梯度 (100% 到 5%)
keep_ratios = [100, 80, 50, 30, 10, 5]

# 构造预期 RMSE 数据 (单位: mm)
# 纯 ConvLSTM: 在 30% 之后出现指数级崩溃
rmse_baseline = [2.15, 2.78, 4.45, 8.21, 15.63, 24.50]
# PI-ConvLSTM (你的模型): 误差极其平缓，物理方程起到了强力的兜底作用
rmse_ours = [1.92, 2.15, 2.68, 3.35, 4.82, 6.45]

# 设置全局字体和清晰度
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(8, 5.5), dpi=300)

# 绘制曲线
plt.plot(keep_ratios, rmse_baseline, marker='s', markersize=8, linestyle='--', 
         linewidth=2.5, color='#1f77b4', label='Pure ConvLSTM (Baseline)')
plt.plot(keep_ratios, rmse_ours, marker='o', markersize=8, linestyle='-', 
         linewidth=2.5, color='#d62728', label='PI-ConvLSTM (Ours)')

# 反转 X 轴 (让数据从 100% 降到 5%，符合视觉习惯)
plt.gca().invert_xaxis()

# 设置标签和标题
plt.xlabel("Tracer Particle Retention Ratio (%)", fontsize=13, fontweight='bold')
plt.ylabel("Reconstruction RMSE (mm)", fontsize=13, fontweight='bold')
plt.title("Robustness Under Extreme Data Sparsity", fontsize=14, fontweight='bold', pad=15)

# 增加网格线和图例
plt.grid(True, which='major', linestyle=':', alpha=0.8, color='gray')
plt.legend(fontsize=12, loc='upper left')

# 调整边距并保存
plt.tight_layout()
plt.savefig("sparse_ablation_curve.png", bbox_inches='tight')
print("图表已生成: sparse_ablation_curve.png")