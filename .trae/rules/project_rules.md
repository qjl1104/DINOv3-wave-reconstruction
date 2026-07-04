# PI-DINOv3 项目规则

## 项目概述
PI-DINOv3：基于物理信息的稀疏双目立体匹配水波表面重建系统。
从双目相机拍摄的水面图像中重建 3D 波面，结合 DINOv3 视觉基础模型 + Transformer 交叉视图匹配 + PINN 物理约束。

## 技术栈
- Python 3.10+, PyTorch 2.6+, CUDA
- DINOv3 ViT-B/16 (frozen, 86M params, 768 维特征, patch_size=16, RoPE 位置编码)
- HuggingFace transformers 加载 backbone
- OpenCV blob 检测器做稀疏关键点
- PINN 自监督训练（无 ground truth 深度标签）

## 已知波参数（Ground Truth）
- 波高 H = 40mm（双振幅）
- 频率 f = 0.79Hz
- 波长 λ ≈ 2500mm（由散关系 ω² = gk 推算）

## 核心代码文件（`DINOv3/`）
| 文件 | 职责 |
|------|------|
| `config.py` | 统一配置 dataclass，所有超参数 |
| `models.py` | CorrMatchingStereoModel（blob检测器 + DINOv3特征器 + 1D相关体 + soft-argmax） |
| `losses.py` | PINNPhysicsLoss（光度 + 视差正则 + 平滑 + 斜率 + 零均值） |
| `dataset.py` | RectifiedWaveStereoDataset（自动矫正、90/10拆分、特征缓存） |
|- `train.py` | 训练脚本（bf16 AMP + CosineAnnealing + 梯度累积 + 验证） |
|- `inference.py` | 单帧推理 → 3D 点云 + 波拟合 + 可视化（bf16 AMP） |
| `evaluate.py` | 全验证集量化评估（极线误差、物理精度、LR一致性、时序、速度、NCC对比） |
| `temporal_inference.py` | 多帧时序分析（全局旋转 + 逐帧波高/波长 + 频域分析） |
| `precompute_cache.py` | DINOv3 特征 + blob 关键点预计算缓存 |
| `utils.py` | 共享工具（pad_to_patch_size, reproject_to_3d 等） |
| `ablation_sparse.py` | 稀疏度消融：PI-DINOv3 vs NCC |
| `diagnose_model.py` | 6 面板诊断图（极线误差、视差分布、NCC 对比等） |

## 代码风格
- 中文注释
- 配置统一用 `config.py` 的 `Config` dataclass，不硬编码路径和超参数
- 模型定义在 `models.py`，损失在 `losses.py`，数据集在 `dataset.py`
- 新脚本放在 `DINOv3/` 下，历史版本归档到 `DINOv3/_archive/`
- 不要写英文注释，用中文

## 训练流程
1. 先 `python precompute_cache.py` 预计算 DINOv3 特征（训练加速）
2. `python train.py` 训练，输出到 `training_runs/<timestamp>/`
3. 每 5 epoch 验证一次，保存 best_model.pth
4. 训练日志：`training_log.json`，loss 曲线：`loss_history.png`

## 数据路径
- 左图：`data/left_images/`
- 右图：`data/right_images/`
- 标定文件：`DINOv3/1128/paper_params_recalculated.npz`
- DINOv3 模型：`DINOv3/dinov3-base-model/`
- 特征缓存：`DINOv3/feature_cache/`

## 已知问题与改进方向
1. **匹配精度**：soft-argmax 对多峰分布不鲁棒 → 考虑 Sinkhorn 匹配
2. **速度**：✅ 已加 torch.compile（train + inference）
3. **3D 表面噪声**：当前用纯统计滤波（IQR + RANSAC 平面） → 考虑物理一致性滤波
4. **评估不足**：✅ 已创建 `evaluate.py`（全验证集量化评估）
5. **代码注释误导**：部分注释写 "DINOv2" 但实际是 DINOv3

## 评估标准（拟建）
- 极线误差（均值 < 0.5px, <1px 占比 > 90%）
- 物理精度（波高 MAE vs GT 40mm, 波长 MAE vs GT 2500mm）
- Left-Right 一致性（<1px 内点率 > 80%）
- 时序一致性（帧间 jitter）
- 推理速度（ms/frame）
- 与 NCC baseline 对比

## 硬件
- RTX 5080 16GB 显存
- 当前 batch_size=4，bf16 AMP，显存充裕