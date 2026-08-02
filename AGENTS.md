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

**当前基线**：3D 975 段/29.2 万点；主峰率 94%；c=2008mm/s(+1.6%, CI[2007,2008])；
PINN 留出 R²=**0.964**；中心振幅 39.0mm/理论 40；f=0.797Hz；λ=2522mm。
旧基线备份：`wave_modeling/real_run/_baseline_v1/`。

## 交互调试台

`.venv_fs/Scripts/python.exe -m streamlit run inspect_app.py` → localhost:8501。
9 页签逐工序检视；检测页 v1/v2/v3 切换+水沫标记视图；②④⑤有帧播放（全分辨率）。
预渲染 mp4 可得真 50fps。

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

## 数据文件

- pkl/npz 不进 git；重跑前备份 `data/trajectories/` 与 `wave_modeling/real_run/`。
- canonical：detections_*_v3.pkl(+_meta)、trajectories_2d_*_v3nf(+_jumpcut)、
  desc_v3nftracks_*、trajectories_3d_v3nf_hung_dino.pkl。
- `.ref_extract/` 是论文提取与调试图像的暂存区（大二进制，勿提交 git）。

## 环境

`.venv_fs/Scripts/python.exe`（全部依赖已装：torch+cu128, cv2, streamlit, plotly）。
打印中文/特殊符号时设 `PYTHONIOENCODING=utf-8`（GBK 控制台会炸）。
