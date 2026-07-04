# PI-DINOv3：基于物理信息的稀疏双目立体匹配水波表面重建

从双目相机拍摄的水面图像中重建 3D 波面。核心方法：**DINOv3 视觉基础模型特征** + **1D 相关体匹配** + **Sinkhorn 最优传输** + **PINN 物理约束自监督训练**。在水面这种纹理重复、传统 NCC 模板匹配容易产生周期性混叠的场景下，借助 DINOv3 的语义特征与全局 Sinkhorn 一致性约束实现鲁棒匹配。

---

## 方法概览

```
                  ┌──────────────────────────────────────────────────────────┐
                  │              PI-DINOv3 立体匹配流水线                     │
                  └──────────────────────────────────────────────────────────┘

  矫正左图 ──┬──► Blob 检测器 ──► 左关键点 (N×2) + 分数 (N)
             │
             └──► DINOv3 Backbone ──► 左稠密特征图 [768, H/16, W/16]
                    (frozen, bf16)            │
                                              ▼
                                      proj_l (1×1 conv, 768→128)
                                              │
                                              ▼  按行抽取关键点描述子
                                          (n_kp, 128)  ← L2 归一化
                                              │
                                              ▼
  矫正右图 ──┬──► Blob 检测器 ──► 右关键点       │
             │                    (仅用于 LR 一致性评估)│
             │                              │
             └──► DINOv3 Backbone ──► 右稠密特征图    │
                                              │
                                              ▼
                                      proj_r (1×1 conv, 768→128)
                                              │
                                              ▼  按相同 patch 行抽取
                                          (Wf, 128)  ← L2 归一化
                                              │
                                              ▼
                          1D 相关体 corr = left_desc @ right_row.T   (n_kp × Wf)
                                              │
                                              ▼
                          × temperature  →  CorrRefinementNet (1D conv 残差)
                                              │
                                              ▼
                       ┌──────────────────────┴──────────────────────┐
                       │ n_kp == 1                                   │ n_kp > 1
                       ▼                                             ▼
                    softmax(corr)                          Sinkhorn(−corr, ε, 10 iters)
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              ▼
                          soft-argmax → 期望右图列坐标 expected_col
                                              │
                                              ▼
                       disp_pixel = (left_col − expected_col) × patch_size
                                              │
                                              ▼
                       Q 矩阵反投影 → 3D 点云 → RANSAC 平面拟合 → 波高 / 波长
```

### 关键设计

| 模块 | 实现 | 代码位置 |
|------|------|---------|
| 关键点检测 | OpenCV SimpleBlobDetector（按 size 取 top-K） | [models.py](models.py) `SparseKeypointDetector` |
| 特征提取 | HuggingFace DINOv3 ViT-B/16，冻结，bf16 推理 | [models.py](models.py) `DINOv3FeatureExtractor.forward_dense` |
| 特征投影 | 两个**独立**的 1×1 conv (768→128→128, GELU) | [models.py](models.py) `proj_l` / `proj_r` |
| 相关体 | 1D：左关键点描述子 × 右图同 patch 行所有列 | [models.py](models.py) `compute_correlation_at_keypoints` |
| 相关精修 | 3 层 1D conv (1→32→32→1, k=5, GELU) 残差 | [models.py](models.py) `CorrRefinementNet` |
| 匹配分配 | 单点 softmax / 多点 Sinkhorn（对数域 fp32，10 次迭代） | [models.py](models.py) `sinkhorn` |
| 视差回归 | soft-argmax 取期望列坐标 | [models.py](models.py) `compute_correlation_at_keypoints` |
| 3D 重建 | `disparity_to_3d`：`[x,y,disp,1] @ Qᵀ` 后齐次化 | [losses.py](losses.py) `disparity_to_3d` |

### 训练损失（自监督，无 GT 视差标签）

总损失 = $w_1 L_{\text{photo}} + w_2 L_{\text{disp}} + w_3 L_{\text{smooth}} + w_4 L_{\text{slope}} + w_5 L_{\text{zeromean}}$

| 损失项 | 权重 | 计算方式 | 代码位置 |
|------|------|---------|---------|
| **光度损失** `L_photo` | 1.0 | 11×11 patch 加权 L1 + 中心像素 smooth-L1（仅亮关键点） | [losses.py](losses.py) `soft_photometric_loss` + `intensity_penalty` |
| **视差正则** `L_disp` | 2.0 | `F.relu(-disp) × 0.1`，惩罚负视差 | [losses.py](losses.py) `compute_photometric` |
| **平滑损失** `L_smooth` | 2.0 | 3D 点转米后 KNN(k=5) 邻居均值 smooth-L1 | [losses.py](losses.py) `compute_pinn_loss` |
| **斜率惩罚** `L_slope` | 0.5 | `F.relu(slope − 0.4).mean()`，限制水面陡度 | [losses.py](losses.py) `compute_pinn_loss` |
| **零均值** `L_zeromean` | 0.1 | `|height.mean()|`，锚定平均水位 | [losses.py](losses.py) `compute_pinn_loss` |

**说明**：
- 极线约束由架构隐式强制（匹配只在同一 patch 行内进行），所以损失里没有显式极线项。
- 物理约束目前是**通用空间正则**（KNN 平滑 + 斜率 + 零均值），并未使用已知波参数（H、f、λ）作为先验。
- 光度损失假设左右图对应点亮度相似，对水面镜面高光的适用性有限。

---

## 项目结构

```
DINOv3/
├── config.py                  # 统一配置 @dataclass（所有超参数）
├── models.py                  # 模型定义
│                              #   - SparseKeypointDetector (OpenCV Blob)
│                              #   - DINOv3FeatureExtractor (frozen DINOv3)
│                              #   - CorrRefinementNet (1D conv 残差)
│                              #   - CorrMatchingStereoModel (完整流水线)
├── losses.py                  # PINNPhysicsLoss (光度 + 视差 + 平滑 + 斜率 + 零均值)
├── dataset.py                 # RectifiedWaveStereoDataset (矫正 + 90/10 拆分 + 缓存)
├── train.py                   # 训练脚本 (bf16 AMP + Cosine + 梯度累积 + 验证)
├── inference.py               # 单帧推理 → 3D 点云 + 波拟合 + 可视化
├── evaluate.py                # 全验证集量化评估 (6 项指标 + NCC 对比)
├── temporal_inference.py      # 多帧时序分析 (3 阶段：全局校准 → 逐帧 → 频域)
├── precompute_cache.py        # DINOv3 特征 + blob 关键点预计算缓存
├── ablation_sparse.py         # 稀疏度消融：PI-DINOv3 vs NCC
├── diagnose_model.py          # 6 面板诊断图
├── paper_figure_generator.py  # 论文图生成
├── generate_calibration.py    # 从 MATLAB 标定参数生成 .npz
├── utils.py                   # 共享工具 (pad_to_patch_size, reproject_to_3d 等)
└── _archive/                  # 历史版本归档
    ├── phase1_dense/          #   稠密立体匹配尝试
    ├── phase2_multi_ai/       #   多 AI 助手迭代
    ├── phase3_sparse_iter/    #   稀疏重建迭代
    ├── phase4_pinn_v22/       #   PINN 集成版本
    └── utilities/             #   旧分析工具
```

---

## 数据与标定

### 目录结构

```
wave_reconstruction_project/
├── data/
│   ├── left_images/           # 左图 (例: left_00001.bmp)
│   └── right_images/          # 右图 (例: right_00001.bmp)
└── DINOv3/
    ├── dinov3-base-model/     # DINOv3 ViT-B/16 权重（本地）
    ├── 1128/
    │   └── paper_params_recalculated.npz   # 标定文件
    └── feature_cache/         # 预计算缓存 (*.pt)
```

### 标定文件内容

`paper_params_recalculated.npz` 包含：
- `map1_left`, `map2_left`, `map1_right`, `map2_right` — OpenCV 矫正 remap 表
- `Q` — 4×4 视差到深度反投影矩阵

从 MATLAB 标定参数生成：

```bash
python generate_calibration.py
```

### 已知 GT 波参数（造波机设定，仅用于评估）

| 参数 | 值 | 来源 |
|------|------|------|
| 波高 H | 40 mm | 造波机设定 |
| 频率 f | 0.79 Hz | 造波机设定 |
| 波长 λ | ≈2500 mm | 深水波色散关系 $\lambda = g/(2\pi f^2)$ 推算 |

**注意**：这些参数目前仅出现在 [evaluate.py:417-419](evaluate.py) 和 [temporal_inference.py:235](temporal_inference.py) 用于评估对比，**未进入训练损失**。

---

## 使用方法

### 1. 预计算缓存（训练前必做）

```bash
cd DINOv3
python precompute_cache.py            # 跳过已存在
python precompute_cache.py --force    # 重新生成全部
```

每个 stereo pair 生成一个 `.pt` 文件，包含：
- `feat_left` / `feat_right`：DINOv3 稠密特征图 fp16 `[768, H/16, W/16]`
- `keypoints_left` / `keypoints_right`：blob 关键点 `[N, 2]`
- `scores_left` / `scores_right`：blob size 分数 `[N]`
- `left_gray` / `right_gray` / `mask`：矫正后灰度图 + 阈值 mask（uint8）

### 2. 训练

```bash
python train.py
python train.py --resume path/to/checkpoint.pth   # 从 checkpoint 恢复
```

关键超参数（[config.py](config.py)）：

| 参数 | 默认值 | 说明 |
|------|------|------|
| `LEARNING_RATE` | 2e-4 | AdamW 学习率 |
| `NUM_EPOCHS` | 300 | 总训练轮数 |
| `BATCH_SIZE` | 4 | 批大小 |
| `ACCUMULATION_STEPS` | 1 | 梯度累积步数 |
| `MAX_KEYPOINTS` | 1024 | 每帧最大关键点数 |
| `CORR_PROJ_DIM` | 128 | 投影后特征维 |
| `PATCH_SIZE_PHOTOMETRIC` | 11 | 光度损失 patch 大小 |
| `KNN_K` | 5 | PINN 平滑邻居数 |
| `SLOPE_THRESHOLD` | 0.4 | 斜率阈值（弧度） |
| `MAX_PINN_POINTS` | 2000 | PINN 采样上限（控制显存） |

训练输出到 `training_runs/<timestamp>/`：

```
training_runs/20260704-143000/
├── checkpoints/
│   ├── best_model.pth        # 验证 loss 最低
│   └── model_ep10.pth        # 每 10 epoch
├── logs/
│   └── training_log.json     # 完整 loss 历史
├── vis/
│   └── loss_history.png      # loss 曲线
└── config.json               # 配置快照
```

训练流程：
- `sanity_check()` 自检一次前向
- 每 epoch 训练 + `scheduler.step()`
- 每 5 epoch 验证，若 val_loss 创新低保存 `best_model.pth`
- 每 10 epoch 保存周期 checkpoint
- CUDA 异常自动 skip 并清显存
- bf16 AMP + GradScaler，梯度裁剪 `max_norm=1.0`

### 3. 单帧推理

```bash
python inference.py --checkpoint training_runs/.../checkpoints/best_model.pth \
                    --image_index 0 \
                    --output result.png
```

输出 4 子图：3D 波面点云、侧视 + 正弦拟合、俯视、关键点叠加。

后处理流程：
1. `reproject_to_3d` → 3D 点
2. 深度过滤 `Z ∈ [2000, 15000]`
3. IQR 滤波（2×IQR）
4. RANSAC 平面拟合（`residual_thresh=25`），`wave_height = Y − Y_plane`
5. 正弦拟合 $A\cos(kz + \phi) + c$，初值 `k₀ = 2π/2500`
6. 波高 `H = 2|A|`，波长 `λ = 2π/|k|`

### 4. 全验证集评估

```bash
python evaluate.py --checkpoint path/to/best_model.pth
python evaluate.py --checkpoint path/to/best_model.pth --max_frames 20
```

6 项指标：

| 指标 | 目标 | 计算方式 |
|------|------|---------|
| 极线误差 | mean<0.5px, <1px 占比>90% | `\|y_left − y_right_pred\|` |
| 波高 MAE | vs GT 40mm | `\|H_recon − 40\|` 均值 |
| 波长 MAE | vs GT 2500mm | `\|λ_recon − 2500\|` 均值 |
| LR 一致性 | <1px 内点率>80% | 预测右点与实际右点最近邻距离 |
| 时序一致性 | 帧间 jitter | 视差/波高标准差 |
| 推理速度 | ms/frame | CUDA synchronize 计时 |
| NCC baseline 对比 | — | 同关键点上 `cv2.matchTemplate` TM_CCOEFF_NORMED |

### 5. 时序分析

```bash
python temporal_inference.py --model_path path/to/best_model.pth \
                             --limit 500 \
                             --output_csv wave_data.csv
```

3 阶段：
- **Phase 1**：采样 20 帧做全局水面参考平面校准（RANSAC 旋转对齐）
- **Phase 2**：逐帧重投影、旋转到世界坐标、拟合 2D 余弦波
- **Phase 3**：波高/波长分布统计 + 频域分析（与色散关系理论频率对比）

### 6. 稀疏度消融

```bash
python ablation_sparse.py --checkpoint path/to/best_model.pth \
                          --ratios 1.0,0.8,0.5,0.3,0.1,0.05
```

通过 `downsample_keypoints` 随机丢弃关键点，对比 PI-DINOv3 与 NCC 在不同稀疏度下的极线误差、视差稳定性、有效匹配率。

---

## 关键设计说明

### 为什么稀疏而不是稠密？

水面纹理重复且呈正弦分布，稠密立体匹配（SGM、RAFT-Stereo）容易产生**周期性混叠**——多个波峰给出等价匹配。用 blob 检测稀疏关键点（示踪粒子或波峰亮点），让模型聚焦于可区分特征，再借助全局 Sinkhorn 一致性消歧。

### 为什么用 DINOv3（冻结）？

DINOv3 自监督特征具有语义和空间一致性，能在像素亮度相近时区分波峰与波谷。冻结 backbone 避免小数据集过拟合，可训练参数仅 `proj_l` / `proj_r` / `CorrRefinementNet` / `temperature` / `sinkhorn_eps`（约 1M 量级，远小于 backbone 86M）。

### 为什么用 Sinkhorn？

逐点独立 soft-argmax 对多峰分布不鲁棒——多个相同关键点可能同时匹配到同一右图位置。Sinkhorn 最优传输在行内做全局软分配，强制近似一对一匹配，打破退化。当行内仅 1 个关键点时退化为 softmax。

### 训练 vs 推理差异

训练和推理走同一前向路径（无 separate inference 分支）。匹配始终在同一 patch 行内进行（架构级极线约束），无显式 epipolar mask。

---

## 硬件与依赖

- Python 3.10+, PyTorch 2.6+ (CUDA)
- RTX 5080 16GB 显存（当前 batch_size=4, bf16 AMP，显存充裕）
- transformers (HuggingFace), opencv-python, scipy, scikit-learn, matplotlib, tqdm

```bash
pip install torch torchvision transformers opencv-python scipy scikit-learn matplotlib tqdm
```

---

## 已知问题与改进方向

1. **PINN 名实不副**：当前物理约束是通用空间正则（KNN 平滑 + 斜率 + 零均值），未使用已知波参数（H=40mm, f=0.79Hz, λ=2500mm）作为先验。
2. **无时序信息**：模型逐帧独立处理，未利用波动方程时空 PDE 约束。`temporal_inference.py` 仅做事后分析。
3. **左右投影不共享权重**：`proj_l` / `proj_r` 是两个独立 conv，标准立体匹配应共享权重以保证左右特征空间一致。
4. **右图关键点未用于训练**：仅用于评估时的 LR 一致性检查。
5. **Sinkhorn 假设可能过强**：强制近似双随机分配，未处理遮挡；可考虑 unbalanced OT。
6. **光度损失水面适用性**：镜面高光视角相关，左右图对应点亮度可能差异较大；`bright_threshold` 滤暗点保留亮点，恰保留高光区。
7. **训练/验证随机切分**：[dataset.py:53-55](dataset.py) 用 `np.arange + 90/10` 随机切分，未做时序切分，验证指标可能因帧间相关性虚高。
8. **稀疏度信息损失**：1024 关键点在 2560×1600 图上极稀疏，高频波信息可能在匹配阶段就丢失。
