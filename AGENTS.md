# AGENTS.md — 波浪三维重建项目

> 给接手者的速览。详细技术论证见 `README.md`（含全部判决记录与数字）。

## 当前生产配方（2026-07-30，v3 时代）

`run_full_pipeline.py` 一键驱动，mtime 自动跳过未过期环节，`--force` 全量重建：

```
v3 检测(02c_detection_v3_localmax.py: 白顶帽+局部极大值+分水岭, 零形状先验)
→ 外观跟踪(03b_tracker_appearance.py: 12px紧门 + NCC≥0.55重捕, --no-foam 去水沫)
→ 跳切清洗 → DINO 描述子(矫正系采样) → 匈牙利一对一匹配(--hung-only)
→ 互谱裁判(eval_tracks.py) → PINN(run_real_pinn.py) → final_visualize.py
```

**当前基线**（2026-08-04 口径修正后）：3D 975 段/29.2 万点；主峰率 94%；
c=**1992mm/s**（片段级 95% CI [1982,2002]，±10；与实测 0.783Hz 线性深水色散
1993 偏差仅 0.1%；旧口径 2008mm/s 系频率口径混搭伪影，见下「基线可信度审计」）；
PINN 留出 R²=**0.964**（插值技能，无外推能力）；中心振幅 39.0mm/理论 40；
f=0.783Hz（本窗口实测，名义 0.79）；λ=2543mm（色散@0.783Hz 预测 2544）。
旧基线备份：`wave_modeling/real_run/_baseline_v1/`。

## 交互调试台

`.venv_fs/Scripts/python.exe -m streamlit run inspect_app.py` → localhost:8501。
9 页签逐工序检视；检测页 v1/v2/v3 切换+水沫标记视图；②④⑤有帧播放（全分辨率）。
预渲染 mp4 可得真 50fps。

- **稀疏性能边界（2026-08-04，两种抽稀模型，图 thin_traj_density_sweep.png）**：
  ① 物理稀疏（整轨迹抽稀，有效双视密度 q=p²）：**q=0.01（开题 1% 目标线）
  链路仍产出正确 c 与振幅**（25 段/主峰率 90%/c=1990[1820,2094]），
  q=0.0025 互谱有效对不足（3 seed 全灭），q≈1e-4 匹配归零；
  η std 全程稳在 ~29mm。② 可见性（逐帧 Bernoulli 抽稀）：p=0.5 产量即
  -94%（轨迹碎裂、主峰率 60%），p=0.25 匹配归零。**结论：稀疏极限由
  轨迹完整性决定而非瞬时密度；跟踪器（12px 紧门不耐缺帧）是稀疏工况
  瓶颈，缺帧容忍/桥接关联是明确的改进点**。论文中两个口径须分开陈述。
  脚本 thin_tracks_density_sweep.py / thin_density_sweep.py（均幂等可续跑）。

## 关键认知（勿重蹈覆辙）

- **形状先验物理不成立**：标识物圆片成像为椭圆（轴比中位 0.68），圆度/凸度/
  惯性比过滤误伤大量真标识物（v1 召回仅 ~31%）。亮度判据也被推翻（水沫与
  暗标识物分布重叠）。可用判据：连通域面积（≥1500px²=水沫团，精度100%）、
  带内外观分类（n_peaks+边缘梯度，LOO 90.5%）。
- **纯距离关联在高密度下无可行窗口**：关联门打在 EKF 预测位置上（误差~40px），
  而泡沫间距 ~15-29px。dt80 碎裂、dt40 右崩、dt20 雪崩——外观辅助重捕是正解。
- **轨迹间偏移是逐轨迹随机**（非空间场，偏差场假说已证伪），逐片段中位数
  debias 是正确处理。
- **天然水沫必须剔除**（用户明确要求）：变形、质心漂移，不是可靠拉格朗日标记。
  但注意 v3 检测 pkl 保留水沫点（仅打 big_comp 标记），剔除发生在跟踪 --no-foam。
- **坐标系**：canonical 轨迹/检测都在矫正系；DINO 描述子也必须在矫正图采样
  （已修复，旧版在原始图采样错位 10-24px）。
- **预处理右图被 CLAHE 毁掉**（纹理放大）；`data/preprocessed_92/` 形状过滤
  过净不可用；`data/lresult|rresult/` 是师兄处理的同场实验图但水沫未去、
  坐标系不一致——磁盘上不存在现成的去沫图像。
- 浪高仪数据：师兄论文只有统计值（`data/reference/yan2021_gauge/`，已数字化
  图4-10/4-11 曲线）；浪高仪当时在拍摄区域外且不同步，只能比位置无关统计量
  或互相关反演位置，**禁止相位级时程硬对比**。
- 全链路唯一裁判：`wave_modeling/eval_tracks.py`（η std/主峰率/互谱 c+CI），
  任何改动以其数字为准。

## 基线可信度审计（2026-08-04）

三项诊断（新脚本，未改 canonical）结论：

- **c 的诚实 CI ≈ ±10mm/s**（片段级 cluster bootstrap，95% CI [1997,2017]），
  原对级 bootstrap CI[2007,2008] 把有效样本量高估约 200 倍。理论值 1976 落在
  诚实 CI 之外 → +1.6% 是系统偏差（标定尺度/fps/f_meas/物理），非采样噪声。
  基线应表述为「c = 2008 ± 10 mm/s（统计），系统偏差另计」。
  脚本 `wave_modeling/eval_c_block_bootstrap.py`。
- **debias 平反**：短片段（L=20/30 帧）个体振幅被压 44%/23%，但 ≥63 帧片段
  贡献 91% 的点，聚合 η std 仅差 ≈0.8%；min_len 20→100 基线指标无实质变化，
  **不建议**为此提高 min_len；但逐片段振幅/相位分析（含 PINN 训练点筛选）
  应只用 ≥63 帧段。**η std 29mm vs 理论 40mm 的缺口与 debias 无关，原因待查**
  （候选：跟踪平滑、波场空间不均匀）。脚本 `wave_modeling/diag_debias_synthetic.py`。
- **PINN：泄漏≈0，但无时间外推能力**。方向先验改纯 train 估计 R² 不变
  （0.9642，四位有效数字相同），泄漏指控不成立；但前 80% 时段训练→后 20%
  预测 R²=-1.21（反相位崩溃）。**R²=0.964 度量的是同时段时空插值技能，
  论文中不得表述为预测/外推能力**。日志 `wave_modeling/real_run/pinn_honest.log`。
- **c 的 +1.6% 系统偏差已查明（2026-08-04，diag_c_bias_budget.py）**：
  纯口径伪影，非链路缺陷。分解：①理论值 1976 按名义 0.79Hz，而本窗口
  实测主频 0.7833Hz（多估计器一致；8192 栅格 argmax 把链路 f_meas 压到
  0.7812，抛物线插值精测 0.7833）贡献 +0.87%；②解缠绕候选周期用名义
  T_WAVE 而非 1/f_meas 贡献 -0.78%。自洽修正后 **c=1992mm/s vs 线性深水
  @实测f=1993，残余 -0.07%**。排除项：水深（方向相反且 7.5m 池即深水）、
  标定尺度（λ 双标定互差<0.4%）、fps（λ 不含时钟且与 0.783Hz 色散吻合）。
  物理实质：造波机本窗口实际产出 ≈0.783Hz。**已修复（2026-08-04）**：
  解缠绕候选周期改 1/f_meas（eval_tracks.py、diag_hovmoller_xcorr.py、
  eval_c_block_bootstrap.py），测频加抛物线插值
  （peak_freq_interp，diag_hovmoller_xcorr.py）；裁判打印增加
  「线性深水@实测f」对比行。run_real_pinn.measure_direction 仍按名义
  口径（仅求方向）；PINN 已用 --c 1992 重训验证（--tag c1992），
  R²=0.964 不变，产物 field_comparison_c1992.png / pinn_real_c1992.pt。

## 数据文件

- pkl/npz 不进 git；重跑前备份 `data/trajectories/` 与 `wave_modeling/real_run/`。
- canonical：detections_*_v3.pkl(+_meta)、trajectories_2d_*_v3nf(+_jumpcut)、
  desc_v3nftracks_*、trajectories_3d_v3nf_hung_dino.pkl。
- `.ref_extract/` 是论文提取与调试图像的暂存区（大二进制，勿提交 git）。

## 环境

`.venv_fs/Scripts/python.exe`（全部依赖已装：torch+cu128, cv2, streamlit, plotly）。
打印中文/特殊符号时设 `PYTHONIOENCODING=utf-8`（GBK 控制台会炸）。
