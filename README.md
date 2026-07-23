# 波浪三维重建项目（Wave Reconstruction Project）

基于**双目立体视觉 + 示踪粒子跟踪 + DINOv3 语义特征 + 物理约束神经网络（PINN）**
的自由表面波场重建系统：从两台相机拍摄的水面粒子视频，重建连续波面高度场 η(x, y, t)。

**生产链**（`run_full_pipeline.py` 一键端到端，约 1 分钟）：

```
原始图像 (left/right, 1000帧@50Hz)
  → 预处理 + blob 检测（亚像素质心）
  → 单相机 KLT+EKF 时序跟踪（2D 轨迹，canonical）
  → 跨相机匹配（几何判据 dy≈0+视差近恒定，DINOv3 描述子验证/救回）
  → 三角测量（3D 轨迹片段 47 条 / 10003 点）
  → 相干性验证（互谱相位法测 c，与深水理论值 1976 mm/s 对拍）〔数字待重跑〕
  → PINN（互谱测向旋转传播坐标系 + 固定 c=1976，片段级留出评估）〔数字待重跑〕
  → 波成分场（0.6–1.0Hz 带通：f≈0.79Hz, λ=c/f≈2500mm）〔数字待重跑〕
```

> **⚠ 2026-07-22 代码修复说明**：本轮 review 修复了若干影响数值正确性的问题，
> 其中**三角化坐标系混用**（`rematch_rectified.triangulate_pairs`，合成往返中位误差
> ~22m → 修复后 ~2mm，详见第 4 节）意味着现存 `trajectories_3d_v2*.pkl` 及
> `wave_modeling/real_run/*` 全部基于错误绝对几何，需按生产链重新生成
> （重跑前请备份旧 pkl，见"数据保护"节）。本文标注**〔待重跑〕**的数字均测自
> 修复前的数据/方法，仅作量级参考。其余修复：反向视差 clamp 方向化且监督损失改用
> 未 clamp 原始视差（不再掐断梯度）、几何指纹排除查询点自身、checkpoint 键核对与
> 随机初始化告警、描述子采样半 patch 对齐、PINN 留出集改片段级且预处理统计仅拟合
> 训练侧、FFT 前断帧重网格化（短空洞插值/长空洞零填充）、Hann 窗振幅 ×2 修正、
> 波长改用 λ=c/f（原 2443mm 为 FFT 窗 artifact）。

---

## 物理原理总览

整条流水线的物理链条：

```
标定几何（针孔模型 + 对极几何）
   → 粒子示踪（波面的拉格朗日采样）
     → 三角测量（空间射线求交）
       → 波动方程约束的场重建（PINN）
```

水面本身透明、镜面反射、无可匹配纹理，无法直接做立体视觉；
浮性示踪粒子提供可跟踪的纹理点，并假设粒子跟随自由表面运动
（浮力 + 表面张力约束，滑移可忽略），因此粒子位置即波面的离散采样。
注意这给出的是**拉格朗日**数据（跟随粒子的轨迹），而非固定网格上的欧拉场。

**精度链条的物理约束**：本装置 Z≈6–10m，**每 ±1px 的视差/定位误差 ≈
±34mm 深度噪声**（δZ = Z²/(bf)·δd），而波幅只有 40mm——测这个波本质上是
**亚像素精度的游戏**。这一条物理约束判决了所有技术路线的可行性（见"路线判决记录"）。

---

## 生产管线详解

### 1. 相机标定 `camera_calibration/`

物理模型：针孔投影 `x_pixel ~ K·[R|t]·X_world` + 径向/切向畸变。

- 输出内参 K1/K2、畸变系数、外参 R/T，并导出投影矩阵
  P1 = K1·[I|0]、P2 = K2·[R|T] 与基础矩阵 F
- 标定结果：`camera_calibration/params/stereo_calib_params_from_matlab_full.npz`
- 标定重投影误差决定整个几何链的精度上限
- 基线 1413mm，矫正后视差 400–1000px

### 2. 图像预处理与粒子检测 `particle_processing/00–02`

- `00_generate_background.py`：背景建模；`01_preprocess*`：背景减除，消除静止杂波与反光
- `02_particle_detection*.py`：blob 检测（LoG/质心法），需**亚像素级**定位
  （深度误差 ∝ z²/(基线×焦距) × 像素定位误差）

### 3. 单相机 2D 时序跟踪 `particle_processing/03_*`

卡尔曼滤波（匀速/匀加速运动模型）预测 + 匈牙利算法全局分配。
现用参数 max_age=20 / min_hits=5 / dist_thresh=80——**已经实测为甜点**
（实验B：max_age 20→40 后 EKF 滑行把不同粒子续接/合并，去偏 η std 从
40mm 恶化到 84mm，仅 7% 片段见 0.79Hz）。
采样要求（物理约束）：帧率 > 2 × 最高波频（Nyquist）；
粒子帧间位移 < 粒子间平均距离。

### 4. 跨相机轨迹匹配 `particle_processing/`

**几何版：`rematch_rectified.py`**（替代旧 `04_*` 与 `05_*`）。
旧 `04_trajectory_matching.py` 的"左右轨迹起止点像素距离 < 200px"预过滤
在物理上错误（本系统视差 400–1000px，正确匹配被系统性误杀）。
修正版：2D 点先经 `undistortPoints + R1/R2/P1/P2` 到矫正坐标系
（正确匹配必满足 dy≈0、视差为正且近恒定），按绝对帧号对齐比较 +
匈牙利指派，三角化时保留每点帧号；片段级匹配（短片段也接受，
同一粒子的断段给出更多时空采样）。基线输出：46 条 3D 片段 / 9866 点
→ `data/trajectories/trajectories_3d_v2.pkl`。

⚠ **曾踩的坑（2026-07 修复）**：三角化曾把"未矫正的原始系射线"与"矫正系
投影矩阵"混用——`undistortPoints` 未传 R1/R2（射线在原始相机系），却用
`inv(K)@P_rect` 三角化。本装置相机间旋转 ~16.5°，该 bug 的合成往返中位误差达
~22m，真实输出 Z 大量顶在 10000mm 滤波上限。现为标准矫正三角化：
`undistortPoints(R=R1/R2)` → 以 `[I|0]`/`[I|t']` 三角化 → `R1ᵀ` 旋回原始相机系
（合成往返中位误差 ~2mm）。三个 rematch 脚本共用同一实现，
**现存三个 3D pkl 均为修复前产物，需重新生成**。

**DINOv3 增强版（生产）：`rematch_dino_v2.py`**。
在几何判据之上叠加轨迹级 DINO 描述子相似度
（`DINOv3/compute_desc_tracks.py` 在原始图像上采样轨迹点描述子）：
① 独立复核——79 个几何候选对相似度全部 ≥0.717（中位 0.786），
证明几何匹配零错配；② 边界救回——视差波动 30–60px 被几何门拒掉的
候选中救回 4 对 → **47 条片段 / 10003 点**
→ `data/trajectories/trajectories_3d_v2_dino.pkl`
（互谱相干性验证与 v2 同级；c 数值〔待重跑〕）。
该位置只要求轨迹级相似度均值（不要求亚像素精度），
是 DINO 语义特征与经典相位链的正确分工。

旧流程（存档参考）：`04_*` 对极约束 + DTW；`05_reconstruction_3d.py`
用 `cv2.triangulatePoints`（但丢弃帧号）。

### 5. 波动场建模（PINN） `wave_modeling/`

**合成验证：`pinn_v2.py`**（06–09 是未跑通的草稿，保留仅供对照）。
合成行波上端到端验证：场重建网格 relL2 ≈ 2.3%，c 反演误差 <0.1%。

- 控制方程：线性波动方程 η_tt = c²∇²η。**c 不取 √(gh)**——本实验为深水
  （kh≳3.8，见"真值锚点"节），c 由深水色散 c = g/ω 给出；
  可选 Boussinesq 色散修正 +(h²/3)∇²η_tt（逼近完整 Airy 色散
  ω² = gk·tanh(kh) 到 O((kh)²) 阶；h 必须显式传入、单位与坐标一致，
  无默认值以防 mm/m 混用）
- 网络：Fourier 特征（可学习频率）+ tanh MLP，输入内部归一化
- 双重损失：数据损失 + 自归一化 PDE 残差（配置点逐 epoch 重采样）
- 关键实践经验（详见 `pinn_v2.py` 文件头注释）：
  ① Fourier 特征的 sigma 必须贴近且不超过目标谱，过大会逐点背诵不泛化；
  ② 波幅 ~1e-3 m 必须归一化，否则坍缩到 η≡0 平凡解；
  ③ 物理损失需按首值自归一化，固定权重会被二阶导数的 ω² 放大压垮。

**真实数据实验：`run_real_pinn.py`**（生产数据 `trajectories_3d_v2_dino.pkl`）。
流程：PCA 主平面 → 法向残差 η → MAD 离群剔除 → 逐片段去中位数（debias）→
跨片段去重 → **互谱相位法测传播方向并旋转到传播坐标系**
（ξ 沿传播，各向异性 Fourier sigma=(0.5,0.3,8.0)，按各维目标谱周期数推导，
推导注释在 run_real_pinn.py）→ **片段级**留出评估。

- **片段级留出**（38 训练 / 9 测试段，seed 固定；PCA 平面、MAD 阈值、归一化
  bounds 仅拟合训练侧——旧版随机点切分只测插值能力、且上述统计在全池数据上
  拟合存在泄漏，2026-07 已修）。R² 数值〔待重跑〕；c 固定为理论值 1976 mm/s
- **c 反演失败并放弃**：可学习时收敛到 772 mm/s 且持续下滑——数据稀疏下
  噪声的宽带模态把 c 往下拖，反演不可辨识；而互谱相位法从数据独立测得
  c=1975 mm/s → 数据没问题，是 PINN 可辨识性问题。故 c 固定，
  物理损失只承担时空相干性约束。
- 已知系统误差：轨迹间存在 ±300~560mm 常值深度偏差（标定/矫正残差导致，
  与 FoundationStereo 残差实验的"碗形"偏差同源、互相印证），用逐片段
  去中位数工程性扣除（平均水面已知是平的，不损失波动信息）；
  根治手段：重新标定或拍静水参考帧（无需重拍波浪）。
- 另一已知现象：长轨迹原始谱被 0.05–0.08Hz 低频漂移主导——
  相机热胀冷缩导致主点漂移，师兄论文亦记载（建议预热相机）；
  分析时用 [0.5,1.2]Hz 带通取波成分。

### 6. 全链路复现与最终重建 `run_full_pipeline.py` + `final_visualize.py`

`run_full_pipeline.py`（根目录）一键端到端五阶段：
DINO 描述子（缓存按源 2D 轨迹 mtime 失效自动重算，`--force-desc` 强制）→
`rematch_dino_v2.py` → `eval_tracks.py` 相干性 → `run_real_pinn.py` →
`final_visualize.py`。各阶段产出有存在性/非空检查，缺产出立即中止。

最终重建（`wave_modeling/real_run/final_result.png` + `final_field.npz`）：

- 波成分（0.6–1.0Hz 带通）：主频 ≈0.79 Hz；波长改报 **λ = c/f ≈ 2500mm**
  （原"2443mm（理论 2502）"是 FFT 窗 artifact——ξ 孔径仅 ~2422mm、不足一个
  波长，空间 FFT 第一 bin 恰等于孔径长度，不能当作测量）；振幅类数字已经
  Hann 窗增益修正（旧版 2|X|/N 在 Hann 窗下系统性偏小 2 倍）〔待重跑〕
- 原始稠密场在数据稀疏/边缘区含慢变结构（~0.1Hz，rms 可达 100mm，
  源于 debias 把片段部分周期均值映射成空间偏移 + 固定 c 波方程容许
  长波行波解）——分析时用带通波成分，`final_visualize.py` 的
  400mm 覆盖掩模已把外推区留白
- 快速核验：`verify_results.py` → `verify_results.png`
  （最长 3 条轨迹：原始 η(t) 肉眼可见 0.8Hz 正弦骑在慢漂移上、
  带通后主峰回 ~0.8Hz、PINN 波成分逐段 R²〔待重跑〕）

---

## 验证体系：互谱相位法（本项目的"判官"）

**原理**：示踪粒子水平位置近似不动，其 η(t) 是 0.79Hz 单频振荡，
相位由沿传播方向的位置决定。对时间重叠充分的轨迹对，在重叠窗口内求
0.79Hz 处**互谱相位差**（arg(A·conj(B))，等效于用全部重叠样本的最优
相位估计）→ 时滞 τ_ij = Δx_ij·n/c，多对加权最小二乘解出相速度大小 c
与传播方向 n。周期模糊（τ±kT）用两遍展开：先近距对（|τ|<T/2 无模糊）
拟合初值，再对全部对选离预测最近的候选时滞重拟合，配 200 次 bootstrap CI
（注意 CI 偏乐观：bootstrap 重采样的轨迹对间强相关——每片段参与数十对，
且解缠绕不确定性未进入重采样）。

**断帧处理（2026-07 修）**：46/47 条片段存在帧空洞（实测最长 24 帧≈0.37 个
波周期）。FFT/带通前先按真实帧号重网格化：≤4 帧空洞线性插值，>4 帧空洞
零填充——窄带单频信号下零填充相位近似无偏，而长空洞线性插值会注入错误
相位（曾使 c 估计漂移 ~5%）。τ 换算用实测主峰频率而非固定 0.79Hz。

**为什么用互谱相位而不用互相关 argmax**：带通后信号接近纯正弦，
余弦平台期导致 argmax 在不同周期间跳变（实测 c 从 2515 漂到 3706 mm/s，
方向轴倒转）；互谱相位用全部样本，是窄带信号的正确工具。

**实现**：`wave_modeling/diag_hovmoller_xcorr.py`（诊断）；
`wave_modeling/eval_tracks.py`（把 η std / 0.79Hz 主峰率 / 相位 c
打包成"同一杆秤"，可评估任何 3D 轨迹 pkl）。

**基线判决**：粒子链 367 个轨迹对的相位差对沿传播方向距离呈干净线性，
**c = 1975 mm/s（95% CI [1867, 2107]）**〔旧方法旧数据：现行断帧重网格化
在当前 pkl 上测得 ~1876 [1790,1978]，1976 处于 CI 上沿；且三角化修复后
空间几何变化，c 需随数据重新生成重新测量〕，与深水理论值 1976 几乎重合，
传播方向沿 PCA 面内 −v 轴（也解释了 Hovmöller (u,t) 面板为何本就不该有
对角条纹——波不沿 u 传播）。这证明全链路（标定→三角化→跟踪→debias）
在空间相干性意义上定量正确，也是后续所有路线判决的标准。

**Hovmöller 显示陷阱（已修）**：旧 `field_comparison.png` 面板看似
"画得太满、静态梯度"，根因是绘图 bug——`fig.colorbar(im2, ax[2])`
把 ax[2] 当作位置参数 `cax`（色条画进指定 axes），整条色带盖住了
Hovmöller 面板；改 `ax=ax[2]` 修复（run_real_pinn.py 内有注释警示）。
细分箱（100mm×0.1s）占用率仅 11.8%，数据稀疏是事实。

---

## 路线判决记录（全部被互谱相位检验定量裁决）

### FoundationStereo 稠密路线（否决）

`foundation_stereo_test/` 完整部署（venv `.venv_fs`）：几何层面成功
（视差 100% 稠密、深度 4–9m 合理、热态 1.6s/对），但 20 帧时序残差实验：
固定系统偏差 ~141mm + 帧间涨落 ~69mm 的噪声地板，比波幅大一个量级——
瓶颈不是像素噪声而是网络在粒子间无纹理区的"假想面"偏差。
稠密路线若要可用需静水参考帧扣偏差 + 域自适应微调。

### PI-DINOv3 学习式匹配（逐帧立体：否决）

`DINOv3/` 完整流水线（冻结 DINOv3 特征 + 1D 相关体 + Sinkhorn +
几何指纹融合，自监督训练，详见其自带 README）。
`batch_pointcloud.py` 全部 1000 帧批量推理（187,729 点，~188 点/帧），
`diag_dinov3_coherence.py` 以 250mm 网格节点当"虚拟浪高仪"做相位诊断：

- **0% 节点的 FFT 主峰落在 0.79±0.1 Hz**（粒子链带通后中位 0.792 Hz）
- 去静态偏差后节点逐帧残差 std = **458mm，是波幅的 11 倍**；
  静态偏差场本身 std 539mm（−1731..+1273mm）
- 0.79 Hz 互谱相位拟合 c = 14.6 m/s——相位无结构，纯噪声

物理根源：soft-argmax 在 16px patch 相关上的视差精度是 patch 量级
（自评 disparity jitter ~7px/帧）→ ±237mm 深度噪声（34mm/px × 7）。
**判决：DINOv3 逐帧点云不能并入 PINN 训练集。**

两轮修复（均实测否决）：
① snap 亚像素细化（`refine_subpixel.py`）：训练头预测位置 97% 不落在
任何右图 blob ±8px 内——粗匹配本身就不可靠，无从细化；
② 描述子最近邻（`match_descriptor_nn.py`，DINO 特征双向互查 +
Lowe ratio + NCC 复核）：~35 匹配/帧、相似度中位 0.774，
15% 节点出现 0.79Hz 主峰（vs 0%），但 0.79Hz 振幅中位仅 3.4mm、
c 拟合仍无结构。深度分布与粒子链一致（中位 6.1m vs 6.3m）——
**近平面表面上，错匹配产生的 3D 点空间上看着合理（其波长评估只差
15% 的原因），但时间不携带真实局部运动**，互谱相位检验正是抓这一点。

### DINOv3 描述子时序跟踪替代 KLT（否决）

`track_descriptor.py`（缓存关键点 + 描述子，匈牙利帧间关联）轨迹量是
KLT 的 10 倍，`match_dino_tracks.py`（匈牙利 1-to-1）得 336 条 3D 片段 /
11.3 万点，去偏 η std≈50mm、38–51% 片段见 0.79Hz 主峰——但**互谱相位
完全无结构**（c 拟合 8.7–34 m/s 均为噪声）。根因是**身份扩散**：轨迹在
相似 blob 间以每次 <8px 的小步随机游走，频率与振幅对（都在 0.79Hz 振荡）
但相位游走（片段内前后半段相位差中位 62°，仅 14% <1rad）。三轮修复
（匈牙利全局指派 / 门控 25→8px 随断档缩放）都只能缓解不能根治：
KLT 锁的是图像块内容本身，描述子在重复纹理上的区分度救不了相位。
**对测波而言相位就是信息本身，故否决。**

### DINOv3 的最终落位（采用）：跨相机匹配的验证与消歧

见"生产管线详解 第 4 节"——轨迹级描述子相似度复核 + 边界救回，
生产数据 `trajectories_3d_v2_dino.pkl` 由此产出。

---

## 实验参数（刘晔恒硕士论文，SJTU 2025，`021010910059.pdf`）

本项目数据与下列参数同出一源（原始图像 2560×1600 = Phantom VEO-E340L
原生分辨率，确证数据来自大水池实验而非同门小水槽论文），作为整条链路的**真值锚点**：

- 场地：SJTU 多功能拖曳水池 300m×16m×7.5m，实验区距造波机 ~70m，拍摄区 3×3m
- 相机：2× Phantom VEO-E340L，**帧率 50 Hz**，单段 2967 帧 = 59.34s
  （本项目数据为其中 1000 帧 = 20s）
- 规则波：**振幅 A = 0.04m，频率 f = 0.79 Hz**，深水色散波长 λ = 2.5017m
  （c_理论 = λf ≈ 1.976 m/s）
- 水深：论文只给出池深 7.5m、未写明实际充水深度；但对 λ=2.5m 的波，
  只要 h≳1.5m（kh≳3.8，tanh(kh)≈0.999）深水近似即成立，
  c_理论 对任何合理充水深度都稳健
- 标记物：圆形聚乙烯泡沫；另有两个波高传感器做对照
- 注意：同门期刊论文（*Spatial-temporal measurement of waves in
  laboratory based on binocular*, Coastal Engineering 177, 2022）是
  **另一套小水槽装置**（14×1×1.2m、水深 0.80m、相机 1280×1024、
  T=0.57–1.14s），参数不可混用

**已完成的真值验证**（无需新实验的替代手段；标注〔待重跑〕者见顶部修复说明）：
1. 长轨迹 η(t) FFT 主峰 0.75–0.81 Hz（带通后中位 ~0.78–0.79 Hz，与 0.79 Hz 一致）、
   振幅估计 21–43mm（与 A=40mm 一致）→ 粒子链路确实测到了论文规则波。
2. 互谱相位法测速 c ≈ 1900 mm/s 量级、CI 覆盖深水理论值 1976
   （见"验证体系"节；具体数值〔待重跑〕）。
3. 重建场波成分：f≈0.79 Hz / λ=c/f≈2500mm〔待重跑〕
   （见"全链路复现"节；原 2443mm 为 FFT 窗 artifact）。

---

## 关键物理假设与误差来源

| 环节 | 假设/误差源 | 控制手段 |
|---|---|---|
| 标定 | 重投影误差 | 多姿态标定板、覆盖整个视场 |
| 采集 | 两相机**硬同步** | 同步触发；不同步会使运动粒子三角测量错位 |
| 采集 | 相机热漂移（0.05–0.08Hz 低频） | 预热相机；分析时带通剔除 |
| 示踪 | 粒子无滑移跟随水面 | 选用中性浮力/浮性粒子、足够小 |
| 检测 | 亚像素定位精度 | LoG + 质心/高斯拟合 |
| 采样 | 帧率 vs 波频（Nyquist）；粒子密度 vs 空间分辨率 | 实验设计 |
| 三角化 | 位置相关深度偏差（±300~560mm"碗形"） | 逐片段 debias；根治需重标定/静水参考帧 |
| PINN | 线性浅水假设 | 波陡大或深水时需换色散/非线性模型 |

---

## 数据保护与复现性警示

**2026-07 修复后的再生成要求**：现存 `trajectories_3d_v2*.pkl` 与
`wave_modeling/real_run/*` 均产出自修复前的三角化（坐标系混用，见第 4 节），
需重跑 `run_full_pipeline.py` 重新生成。这些 pkl **未被 git 追踪，覆盖不可恢复**，
重跑前请先备份（如 `cp x.pkl x.pkl.prefix.bak`）。

**canonical 数据（勿删勿覆盖）**：
`data/trajectories/trajectories_2d_*_optimized.pkl`（2025-08-07 代 2D 轨迹）、
`trajectories_3d_v2.pkl`、`trajectories_3d_v2_dino.pkl`。

当前磁盘上的预处理图像（2025-08-24 生成）与上述 2D 轨迹并非同一代——
在其上用现行 `02_particle_detection.py`（08-03 版）重检 + 默认参数 03 +
rematch，仅得 6 条片段 / 741 点（v2 为 46 条 / 9866 点）。
v2 的 2D 轨迹来自 08-07 代的预处理+检测组合（检测 pkl 已删、
预处理已被覆盖，疑为 `02_particle_detection_visible_opus_9/92.py`
系列——该系列实为"预处理+形状分类"一体脚本；其默认配置输出过净，
KLT 会"饿死"，直接复跑不能复现）。要扩容轨迹需先复现 08-07 代组合。

**工程陷阱（均已踩过并修复）**：
- 三角化：`cv2.undistortPoints` 不传 R 得到的是**原始相机系**归一化射线，
  只能与原始系 [R|t] 配对；与矫正系 P 矩阵混用时，16.5° 相机间旋转造成
  合成往返中位误差 ~22m。正确做法：`undistortPoints(R=R1/R2)` 到矫正系 →
  `[I|0]`/`[I|t']` 三角化 → `R1ᵀ` 旋回（`rematch_rectified.triangulate_pairs`）。
- Hann 窗频谱振幅必须除以窗增益：2|X|/Σw（≈4|X|/N），写 2|X|/N 会系统性
  偏小 2 倍，直接污染振幅真值对比。
- 断帧序列不能直接 rfft（压缩频率轴），也不能跨长空洞线性插值（注入错误
  相位）：≤4 帧插值、>4 帧零填充（各 wave_modeling 脚本的 `_regrid_uniform`）。
- DINOv3 侧：不建 dataset 直接推理时必须先把 `cfg.IMAGE_WIDTH/HEIGHT`
  按标定图尺寸回填（几何指纹要除图像对角线，默认 0 → 除零 →
  全帧视差 NaN，`batch_pointcloud.py` 已处理）。
- 绘图：`fig.colorbar(im2, ax[2])` 的位置参数是 `cax` 不是 `ax`，
  会把色条画进面板盖住数据（run_real_pinn.py 内已改 `ax=ax[2]`）。
- pickle 跨脚本：`Track` 等类按 `__module__` 记录路径，
  importlib 动态加载 03 脚本时需把类的 `__module__` 改为 `__main__`
  并注入命名空间（`run_03_default.py`/`run_03_tuned.py` 已处理）；
  反序列化用 rematch_rectified.py 的同名桩类接管。

---

## 参考文献与开源工具

**立体波面测量（经典方法）**
- Benetazzo (2006), *Measurements of short water waves using stereo matched image sequences*, Coastal Engineering（WASS 系统）
- Wanek & Wu (2006), *Automated trinocular stereo imaging system for 3D surface wave measurements*, Ocean Engineering（ATSIS）
- de Vries et al. (2011), 立体视觉测自由表面波高
- Gallego et al. (2014), *Space-time ocean wave measurement using variational stereo*

**折射/纹影法（单相机测波面坡度，实验室水槽适用）**
- Moisy, Rabaud & Salsac (2009), *A synthetic Schlieren method (FS-SS)*, Experiments in Fluids
- Gomit et al. (2013), *Free surface measurement by stereo-refraction*, Experiments in Fluids

**PINN 与波动建模**
- Raissi et al. (2019), *Physics-informed neural networks*, JCP
- Sallam (2024), *PINN + Fourier features 从水平速度反演波面*, arXiv:2409.19851
- Haitsiukevich & Ilin, *bif-PINN 求解自由表面水波问题*, JCP
- [DeepXDE](https://github.com/lululxvi/deepxde)：PINN 快速实验框架

**开源工具**
- 检测/跟踪：[TrackPy](http://soft-matter.github.io/trackpy/)（Crocker–Grier 亚像素检测 + Crocker–Grier/LAP 链接）
- 特征匹配：[LightGlue](https://github.com/cvg/lightglue)（SuperGlue 的快速继任者）
- 视觉基础模型特征：[DINOv3](https://github.com/facebookresearch/dinov3)（本项目用于匹配验证/消歧）
- 稠密立体匹配：RAFT-Stereo、[IGEV-Stereo](https://github.com/gangweiX/IGEV)、[FoundationStereo](https://github.com/NVlabs/FoundationStereo)（zero-shot，已判决不适用）
