# inspect_app.py
# -*- coding: utf-8 -*-
"""
波浪重建全链路工序调试台（streamlit 网页）。

每个页签对应一道工序，可交互翻看任意帧/任意轨迹，打开黑箱逐环评估：
  总览 | 预处理/矫正 | 检测 | 2D 跟踪 | 跨相机匹配 | 3D 点云 | 互谱裁判 | PINN 重建 | 基准对比

启动（项目根目录）：
    .venv_fs/Scripts/python.exe -m streamlit run inspect_app.py
"""
import os
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np
import streamlit as st
ROOT = os.path.dirname(os.path.abspath(__file__))
PP_DIR = os.path.join(ROOT, "particle_processing")
sys.path.insert(0, PP_DIR)

# 2D 轨迹 pkl 反序列化需要 Track 等类定义（与 stage_gallery.py 相同的补丁）
import __main__  # noqa: E402
import rematch_rectified as rr  # noqa: E402

for _n in ["Track", "UltraTrack", "WaveParticleTrack", "StrictTrack",
           "SimpleKalmanFilter", "ImprovedKalmanFilter",
           "ExtendedKalmanFilter", "OptimizedExtendedKalmanFilter",
           "UltraOptimizedKalmanFilter", "WaveParticleKalmanFilter",
           "StrictWaveKalmanFilter"]:
    setattr(__main__, _n, getattr(rr, _n))
__main__.RobustKalmanFilter = type("RobustKalmanFilter", (rr.SimpleKalmanFilter,), {})

DISP_PLANE = rr.DISP_PLANE
DISP_PLANE_TOL = rr.DISP_PLANE_TOL

CALIB = os.path.join(ROOT, "camera_calibration/params/stereo_calib_params_from_matlab_full.npz")
TRAJ2D = {s: os.path.join(ROOT, f"data/trajectories/trajectories_2d_{s}_v3nf_jumpcut.pkl") for s in ("left", "right")}
TRAJ3D = os.path.join(ROOT, "data/trajectories/trajectories_3d_v3nf_hung_dino.pkl")
PINN_PT = os.path.join(ROOT, "wave_modeling/real_run/pinn_real.pt")
FIELD_NPZ = os.path.join(ROOT, "wave_modeling/real_run/final_field.npz")
REAL_RUN = os.path.join(ROOT, "wave_modeling/real_run")
YAN_DIR = os.path.join(ROOT, "data/reference/yan2021_gauge")
FPS = 50.0
F_WAVE = 0.79
C_THEORY = 1976.0  # mm/s

st.set_page_config(page_title="波浪重建工序调试台", layout="wide")


# ---------------------------------------------------------------- 数据加载（缓存）
@st.cache_resource
def load_calib():
    return np.load(CALIB)


@st.cache_data
def load_detections_meta(side):
    """v3 检测的 meta(area/peak/contrast/big_comp)，无则返回 None。"""
    p = os.path.join(ROOT, f"data/detections/detections_{side}_v3_meta.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_detections(side, version="v1"):
    suffix = "" if version == "v1" else f"_{version}"
    with open(os.path.join(ROOT, f"data/detections/detections_{side}{suffix}.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_tracks2d(side, variant="v3app"):
    with open(os.path.join(ROOT, f"data/trajectories/trajectories_2d_{side}_{variant}_jumpcut.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_traj3d(variant="v3app_hung"):
    with open(os.path.join(ROOT, f"data/trajectories/trajectories_3d_{variant}_dino.pkl"), "rb") as f:
        return pickle.load(f)


TRAJ2D_VARIANTS = {"v3app (含泡沫)": "v3app", "v3nf (泡沫过滤)": "v3nf"}
TRAJ3D_VARIANTS = {"v3app_hung (含泡沫)": "v3app_hung", "v3nf_hung (泡沫过滤)": "v3nf_hung"}


@st.cache_data
def load_field():
    f = np.load(FIELD_NPZ)
    return {k: f[k] for k in f.files}


@st.cache_data
def load_yan():
    out = {}
    p1 = os.path.join(YAN_DIR, "fig4-10_gauge_4s.csv")
    p2 = os.path.join(YAN_DIR, "fig4-11_gauge_vs_binocular_4s.csv")
    if os.path.exists(p1):
        out["gauge410"] = np.loadtxt(p1, delimiter=",", skiprows=1)
    if os.path.exists(p2):
        out["cmp411"] = np.loadtxt(p2, delimiter=",", skiprows=1)
    return out


@st.cache_data
def rect_frame(side, frame0, gray=True):
    """frame0 为 0 基帧号；返回矫正图（gray 或 bgr）。全分辨率。"""
    calib = load_calib()
    img = cv2.imread(os.path.join(ROOT, f"data/{side}_images/{side}{frame0 + 1:04d}.bmp"), 0)
    if img is None:
        return None
    rect = cv2.remap(img, calib[f"map1_{side}"], calib[f"map2_{side}"], cv2.INTER_LINEAR)
    return rect if gray else cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- 小组件
def regrid_fft(fr, eta):
    """缺帧重采样（≤4 帧空洞线性插值，长洞置 0）后做 FFT，返回 (freqs, amp)。"""
    fr = np.asarray(fr)
    grid = np.arange(fr[0], fr[-1] + 1)
    out = np.interp(grid, fr, eta)
    long = np.flatnonzero(np.diff(fr) - 1 > 4)
    for i in long:
        out[fr[i] - grid[0] + 1: fr[i + 1] - grid[0]] = 0.0
    out = out - out.mean()
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / FPS)
    return freqs, spec


def eta_of_traj3d(trajs, plane):
    """3D 轨迹 → (片段列表[(帧, η)], 平面法向, 质心)。"""
    c, nvec = plane
    out = []
    for t in trajs:
        if len(t) < 2:
            continue
        eta = (t[:, 1:4] - c) @ nvec
        out.append((t[:, 0].astype(int), eta))
    return out


def plane_vt(pts):
    """返回与 np.linalg.svd(pts-c) 相同的 vt 行基（方差降序），
    用 3x3 协方差 eigh 实现，避免 (N,3) 全量 SVD 的 (N,N) 内存爆炸。"""
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    _, ev = np.linalg.eigh(cov)  # 列向量，特征值升序
    return c, ev[:, ::-1].T      # 行基降序：[最大方差, 次大, 法向]


def fit_plane(pts):
    c, vt = plane_vt(pts)
    nvec = vt[2] * np.sign(vt[2][2])
    return c, nvec


def stat_card(label, value, help_txt=None):
    st.metric(label, value, help=help_txt)


def frame_player(key, label, values):
    """一体化帧播放器：返回 (frame, playing, fps)。
    播放中滑块隐藏（显示当前帧文本），纯计数键推进——完全不触碰
    控件 session_state，规避新版 streamlit 的控件键修改限制；
    暂停时滑块出现，拖动选择起始帧，再播放从该帧继续。"""
    play_key, ctr_key = f"{key}_playing", f"{key}_ctr"
    if play_key not in st.session_state:
        st.session_state[play_key] = False
    if ctr_key not in st.session_state:
        st.session_state[ctr_key] = 0
    c1, c2 = st.columns([1, 3])
    if c1.button("⏸ 暂停" if st.session_state[play_key] else "▶ 播放", key=f"{key}_btn"):
        was = st.session_state[play_key]
        st.session_state[play_key] = not was
        if was:  # 播放→暂停：标记，下一渲染把控件预置到停下的帧
            st.session_state[f"{key}_justplayed"] = True
        st.rerun()
    fps = c2.selectbox("目标帧/秒（实际受渲染限制）", [1, 2, 5, 10, 20, 50], index=2, key=f"{key}_fps")
    if st.session_state[play_key]:
        frame = values[st.session_state[ctr_key] % len(values)]
        st.caption(f"播放中：frame {frame}（暂停后可拖动跳转）")
    else:
        if st.session_state.get(f"{key}_justplayed"):
            # 实例化前预置（合法时机），并清除标记
            st.session_state[f"{key}_w"] = values[st.session_state[ctr_key] % len(values)]
            st.session_state[f"{key}_justplayed"] = False
        frame = st.select_slider(label, options=values, key=f"{key}_w")
        st.session_state[ctr_key] = values.index(frame)
    return frame, st.session_state[play_key], fps


def frame_player_tick(key, values, fps):
    """页签渲染末尾调用：播放中则等待后推进计数键并 rerun。"""
    if not st.session_state.get(f"{key}_playing"):
        return
    time.sleep(1.0 / fps)
    st.session_state[f"{key}_ctr"] = (st.session_state[f"{key}_ctr"] + 1) % len(values)
    st.rerun()


# ---------------------------------------------------------------- 页签实现
def tab_overview():
    st.header("全链路总览")
    st.markdown("""每道工序的交互检视见上方页签。本页做**产物新鲜度体检**：
若产物的修改时间早于其输入（红色），说明该环节在输入更新后未重跑，看到的图是旧数据。""")

    chain = [
        ("2D 轨迹 jumpcut (左)", TRAJ2D["left"], []),
        ("2D 轨迹 jumpcut (右)", TRAJ2D["right"], []),
        ("DINO 描述子 (左, 生产 v3nf)", os.path.join(ROOT, "DINOv3/desc_v3nftracks_left.pkl"), [TRAJ2D["left"]]),
        ("DINO 描述子 (右, 生产 v3nf)", os.path.join(ROOT, "DINOv3/desc_v3nftracks_right.pkl"), [TRAJ2D["right"]]),
        ("DINO 描述子 (左, 对照 v3app)", os.path.join(ROOT, "DINOv3/desc_v3apptracks_left.pkl"), [TRAJ2D["left"]]),
        ("DINO 描述子 (右, 对照 v3app)", os.path.join(ROOT, "DINOv3/desc_v3apptracks_right.pkl"), [TRAJ2D["right"]]),
        ("3D 轨迹 (v3nf_hung)", TRAJ3D, [TRAJ2D["left"], TRAJ2D["right"]]),
        ("PINN 模型", PINN_PT, [TRAJ3D]),
        ("最终场 npz", FIELD_NPZ, [PINN_PT]),
        ("最终成果图", os.path.join(REAL_RUN, "final_result.png"), [PINN_PT]),
        ("工序画廊 stage4", os.path.join(ROOT, "data/visualization/stage4_matches.png"), [TRAJ3D]),
        ("工序画廊 stage5", os.path.join(ROOT, "data/visualization/stage5_pointcloud.png"), [TRAJ3D, PINN_PT]),
    ]
    rows = []
    for name, path, deps in chain:
        if not os.path.exists(path):
            rows.append((name, "缺失", "-", "-"))
            continue
        mt = os.path.getmtime(path)
        stale = [os.path.basename(d) for d in deps if os.path.exists(d) and os.path.getmtime(d) > mt]
        status = "🔴 过期(早于: " + ", ".join(stale) + ")" if stale else "🟢 最新"
        rows.append((name, time.strftime("%m-%d %H:%M", time.localtime(mt)),
                     f"{os.path.getsize(path) / 1e6:.1f} MB", status))
    st.table({"产物": [r[0] for r in rows], "修改时间": [r[1] for r in rows],
              "大小": [r[2] for r in rows], "状态": [r[3] for r in rows]})

    st.subheader("关键数量")
    try:
        trajs = load_traj3d()
        n_pts = sum(len(t) for t in trajs)
        fr_all = np.concatenate([t[:, 0] for t in trajs if len(t)])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("3D 轨迹段数", len(trajs))
        c2.metric("总 3D 点数", n_pts)
        c3.metric("帧覆盖范围", f"{int(fr_all.min())}–{int(fr_all.max())}")
        c4.metric("平均每帧点数", f"{n_pts / max(1, len(np.unique(fr_all))):.0f}")
    except Exception as e:
        st.warning(f"3D 轨迹读取失败: {e}")

    st.subheader("刷新命令速查")
    st.code("""# 全链路重跑（轨迹→匹配→裁判→PINN→最终图）
.venv_fs/Scripts/python.exe run_full_pipeline.py
# 5 张静态工序图
.venv_fs/Scripts/python.exe particle_processing/stage_gallery.py
# 2D 轨迹质检三件套 / 动态视频
.venv_fs/Scripts/python.exe particle_processing/qc_tracks_2d.py left
.venv_fs/Scripts/python.exe particle_processing/qc_tracks_video.py left
# 互谱裁判（Hovmöller + 独立测 c）
.venv_fs/Scripts/python.exe wave_modeling/diag_hovmoller_xcorr.py
# 一眼核验
.venv_fs/Scripts/python.exe wave_modeling/verify_results.py""", language="bash")


def tab_preprocess():
    st.header("① 预处理与矫正")
    side = st.radio("相机", ["left", "right"], horizontal=True)
    frame0, playing, fps = frame_player("pp", "帧号(0基)", list(range(1000)))
    raw = cv2.imread(os.path.join(ROOT, f"data/{side}_images/{side}{frame0 + 1:04d}.bmp"), 0)
    rect = rect_frame(side, frame0)
    pp_path = os.path.join(ROOT, f"data/preprocessed/{side}/preprocessed_frame_{frame0:05d}.png")
    pp = cv2.imread(pp_path, 0) if os.path.exists(pp_path) else None
    imgs, titles = [], []
    if raw is not None:
        imgs.append(raw); titles.append("原始图")
    if rect is not None:
        imgs.append(rect); titles.append("矫正图")
    if pp is not None:
        imgs.append(pp); titles.append("预处理后(背景减除)")
    if imgs:
        st.image(np.hstack(imgs), caption=" | ".join(titles), width='stretch')
    bg_path = os.path.join(ROOT, f"data/preprocessed/background_{side}.png")
    if os.path.exists(bg_path):
        with st.expander("背景模型图"):
            st.image(cv2.imread(bg_path, 0), width='stretch')
    st.caption("看点：预处理后泡沫是否清晰保留、背景是否压干净；矫正后左右图同一行是否对应同一水平线。")
    frame_player_tick("pp", list(range(1000)), fps)


def tab_detection():
    st.header("② 粒子检测")
    col_s, col_v = st.columns(2)
    side = col_s.radio("相机", ["left", "right"], horizontal=True, key="det_side")
    version = col_v.radio("检测版本", ["v1", "v2", "v3"], horizontal=True, key="det_ver",
                          help="v1=生产严格形状过滤 | v2=放宽形状过滤 | v3=局部极大值(零形状先验)")
    dets = load_detections(side, version)
    metas = load_detections_meta(side) if version == "v3" else None
    frame0, playing, fps = frame_player("det", "帧号(0基)", list(range(len(dets))))
    hide_foam = False
    if version == "v3" and metas is not None:
        hide_foam = st.checkbox("隐藏水沫（big_comp≥1，等同生产 --no-foam 视图）", value=True)
    rect = rect_frame(side, frame0, gray=False)
    pts = dets[frame0]
    mm = metas[frame0] if metas is not None else None
    n_clean, n_foam = 0, 0
    for k, (x, y) in enumerate(pts):
        big = int(mm[k][3]) if (mm is not None and len(mm) > k and len(mm[k]) >= 4) else 0
        if big >= 1:
            n_foam += 1
            if hide_foam:
                continue
            col = (0, 0, 255) if big == 1 else (0, 128, 255)  # 红=巨型水沫域 橙=带内水沫
        else:
            n_clean += 1
            col = (0, 255, 0)
        cv2.circle(rect, (int(x), int(y)), 8, col, 2, cv2.LINE_AA)
    cv2.putText(rect, f"{n_clean} markers + {n_foam} foam @f{frame0} [{version}]", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    st.image(bgr2rgb(rect), caption=f"检测点叠加 frame {frame0} ({version})：绿=标识物 红=巨型水沫域 橙=带内水沫", width='stretch')
    counts = np.array([len(d) for d in dets])
    st.line_chart({"每帧检测数": counts})
    st.caption(f"{version} 全序列: 均值 {counts.mean():.0f}, 最小 {counts.min()}, 最大 {counts.max()}。"
               "看点：检测数是否稳定；突然掉零说明预处理/阈值在该帧失效。"
               "（v3 检测 pkl 保留水沫点仅打标记，生产链跟踪时 --no-foam 剔除）")
    frame_player_tick("det", list(range(len(dets))), fps)

    with st.expander("真 50fps 播放：预渲染视频片段"):
        st.caption("逐帧 rerun 的播放方式受渲染/传输限制（实际 ~2-5fps）。"
                   "预渲染 mp4 才是满帧率播放。对当前相机+版本+水沫开关渲染。")
        n_seg = st.number_input("片段长度（帧）", 50, 1000, 200, 50, key="det_seglen")
        if st.button("渲染并播放", key="det_vid_btn"):
            vpath = os.path.join(ROOT, f"data/visualization/det_preview_{side}_{version}.mp4")
            with st.spinner(f"渲染 {n_seg} 帧..."):
                vw = None
                for f0 in range(min(int(n_seg), len(dets))):
                    frm = rect_frame(side, f0, gray=False)
                    pm = metas[f0] if metas is not None else None
                    for k, (x, y) in enumerate(dets[f0]):
                        big = int(pm[k][3]) if (pm is not None and len(pm) > k and len(pm[k]) >= 4) else 0
                        if big >= 1:
                            if hide_foam:
                                continue
                            col = (0, 0, 255) if big == 1 else (0, 128, 255)
                        else:
                            col = (0, 255, 0)
                        cv2.circle(frm, (int(x), int(y)), 8, col, 2, cv2.LINE_AA)
                    if vw is None:
                        vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 50,
                                             (frm.shape[1], frm.shape[0]))
                    vw.write(frm)
                vw.release()
            st.video(vpath)


def tab_tracks2d():
    st.header("③ 2D 轨迹跟踪")
    col_s, col_v = st.columns(2)
    side = col_s.radio("相机", ["left", "right"], horizontal=True, key="t2_side")
    variant = col_v.radio("数据集", list(TRAJ2D_VARIANTS.keys()), horizontal=True, key="t2_var")
    try:
        tracks = load_tracks2d(side, TRAJ2D_VARIANTS[variant])
    except FileNotFoundError:
        st.warning(f"数据集 {variant} 的轨迹文件尚未生成（泡沫过滤链可能还在跑）")
        return
    st.write(f"轨迹总数: **{len(tracks)}**（跳切清洗版，{variant}）")

    lens = np.array([len(t.points) for t in tracks])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("轨迹长度分布")
        st.bar_chart(np.histogram(lens, bins=30)[0])
    with col2:
        st.markdown("叠加图（随机染色，最多 400 条）")
        max_show = st.slider("显示条数", 50, len(tracks), min(400, len(tracks)))
        rect = rect_frame(side, 0, gray=False)
        rng = np.random.default_rng(0)
        for t in tracks[:max_show]:
            fr = sorted(t.points)
            pts = np.array([t.points[k] for k in fr], np.int32)
            col = tuple(int(c) for c in rng.integers(60, 255, 3))
            cv2.polylines(rect, [pts], False, col, 1, cv2.LINE_AA)
        st.image(bgr2rgb(rect), caption="点团/短弧=好；游走长线=身份污染", width='stretch')

    st.subheader("单轨迹检查器")
    order = np.argsort(-lens)
    pick = st.selectbox("选一条轨迹（按长度降序）",
                        [f"#{i} 长{lens[i]} 帧{min(tracks[i].points)}–{max(tracks[i].points)}" for i in order[:50]])
    idx = int(pick.split(" ")[0][1:])  # 标签以 "#原始索引" 开头
    t = tracks[idx]
    frs = np.array(sorted(t.points))
    ys = np.array([t.points[k][1] for k in frs])
    xs = np.array([t.points[k][0] for k in frs])
    freqs, spec = regrid_fft(frs, ys - ys.mean())
    band = (freqs > 0.4) & (freqs < 1.5)
    f_pk = freqs[band][np.argmax(spec[band])] if band.any() else 0

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(cols=2, rows=1, subplot_titles=("y(t) 像素坐标", "y(t) FFT 谱"))
    fig.add_trace(go.Scatter(x=frs / FPS, y=ys, mode="lines", name="y"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freqs[band], y=spec[band], mode="lines", name="FFT"), row=1, col=2)
    fig.add_vline(x=F_WAVE, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_xaxes(title_text="t (s)", row=1, col=1)
    fig.update_xaxes(title_text="f (Hz)", row=1, col=2)
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.caption(f"FFT 主峰 {f_pk:.3f} Hz（理论 {F_WAVE} Hz）。主峰对准=轨迹跟住了同一个泡沫的波动。")

    st.markdown("Patch 抽查（沿轨迹寿命等距 8 帧，肉眼确认是否同一泡沫）")
    cols = st.columns(8)
    show_fr = np.linspace(frs[0], frs[-1], 8).astype(int)
    for c, f0 in zip(cols, show_fr):
        if f0 not in t.points:
            c.caption(f"f{f0}\n(缺帧)")
            continue
        img = rect_frame(side, int(f0))
        x, y = t.points[f0]
        x, y = int(x), int(y)
        patch = img[max(0, y - 40):y + 40, max(0, x - 40):x + 40]
        c.image(patch, caption=f"f{f0}", width='stretch')


def tab_matching():
    st.header("④ 跨相机匹配")
    variant = st.radio("数据集", list(TRAJ3D_VARIANTS.keys()), horizontal=True, key="m_var")
    try:
        trajs = load_traj3d(TRAJ3D_VARIANTS[variant])
    except FileNotFoundError:
        st.warning(f"数据集 {variant} 的 3D 轨迹尚未生成")
        return
    calib = load_calib()
    from collections import Counter
    cnt = Counter()
    for t in trajs:
        for fr in set(t[:, 0].astype(int)):
            cnt[fr] += 1
    frames_sorted = [int(f) for f, _ in cnt.most_common()]
    if not frames_sorted:
        st.warning(f"数据集 {variant} 的 3D 轨迹为空，无匹配点可展示")
        return
    frame, playing, fps = frame_player("m", "帧（按匹配点数排序，靠前=点多）", frames_sorted[:200])
    n_pts_frame = cnt[frame]
    st.write(f"frame {frame}: **{n_pts_frame}** 个匹配点")

    K_rect = calib["P1"][:, :3]
    t_rect = np.linalg.inv(calib["P2"][:, :3]) @ calib["P2"][:, 3]
    R1 = calib["R1"]
    pts = np.vstack([t[t[:, 0] == frame, 1:4] for t in trajs if np.any(t[:, 0] == frame)])
    xr = pts @ R1.T
    hl = xr @ K_rect.T
    pl = hl[:, :2] / hl[:, 2:3]
    hr = (xr + t_rect) @ K_rect.T
    pr = hr[:, :2] / hr[:, 2:3]

    L = rect_frame("left", frame, gray=False)
    Rr = rect_frame("right", frame, gray=False)
    both = np.hstack([L, Rr])
    W = L.shape[1]
    n_show = st.slider("显示连线数", 10, min(200, len(pl)), min(60, len(pl)))
    rng = np.random.default_rng(2)
    sel = rng.choice(len(pl), min(n_show, len(pl)), replace=False)
    for k in sel:
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        cv2.line(both, tuple(np.int32(pl[k])), tuple(np.int32(pr[k] + [W, 0])), col, 1, cv2.LINE_AA)
        cv2.circle(both, tuple(np.int32(pl[k])), 4, col, -1, cv2.LINE_AA)
        cv2.circle(both, tuple(np.int32(pr[k] + [W, 0])), 4, col, -1, cv2.LINE_AA)
    st.image(bgr2rgb(both), caption="连线应近似水平（矫正后 dy≈0）", width='stretch')

    dy = pl[:, 1] - pr[:, 1]
    disp = pl[:, 0] - pr[:, 0]
    d_pred = DISP_PLANE[0] * pl[:, 0] + DISP_PLANE[1] * pl[:, 1] + DISP_PLANE[2]
    col1, col2, c3 = st.columns(3)
    col1.metric("|dy| 中位", f"{np.median(np.abs(dy)):.2f} px", help="矫正质量；应 < 3 px")
    col2.metric("视差 std", f"{disp.std():.2f} px", help="深度近似不变 → 视差应近恒定")
    c3.metric("视差-平面预测 偏差中位",
              f"{np.median(disp - d_pred):.2f} px",
              help=f"DISP_PLANE 先验容差 ±{DISP_PLANE_TOL:.0f} px")
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=disp - d_pred, nbinsx=40, name="disp − plane_pred"))
    fig.update_layout(title="视差相对 DISP_PLANE 预测的偏差分布（应集中在 0 附近）",
                      height=300)
    st.plotly_chart(fig, width='stretch')
    frame_player_tick("m", frames_sorted[:200], fps)


def tab_pointcloud():
    st.header("⑤ 3D 点云与波剖面")
    import plotly.graph_objects as go
    variant = st.radio("数据集", list(TRAJ3D_VARIANTS.keys()), horizontal=True, key="pc_var")
    try:
        trajs = load_traj3d(TRAJ3D_VARIANTS[variant])
    except FileNotFoundError:
        st.warning(f"数据集 {variant} 的 3D 轨迹尚未生成")
        return
    mode = st.radio("模式", ["单帧 3D 散点", "全序列点云(抽稀)", "单帧波剖面"], horizontal=True)
    all_fr = np.concatenate([t[:, 0] for t in trajs if len(t)])
    f_min, f_max = int(all_fr.min()), int(all_fr.max())
    if mode != "全序列点云(抽稀)":
        frame0, playing, fps = frame_player("pc", "帧号(0基)", list(range(f_min, f_max + 1)))
        pts = np.vstack([t[t[:, 0] == frame0, 1:4] for t in trajs if np.any(t[:, 0] == frame0)])
        st.write(f"frame {frame0}: {len(pts)} 点")
    else:
        pts_all = np.vstack([t[:, 1:4] for t in trajs if len(t)])
        idx = np.random.default_rng(0).choice(len(pts_all), min(20000, len(pts_all)), replace=False)
        pts = pts_all[idx]
        st.write(f"全部 {len(pts_all)} 点，抽稀显示 {len(pts)}")

    if mode != "单帧波剖面":
        fig = go.Figure(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                                     marker=dict(size=2, color=pts[:, 2], colorscale="Viridis",
                                                 colorbar=dict(title="Z mm"))))
        fig.update_layout(height=600, scene=dict(xaxis_title="X mm", yaxis_title="Y mm",
                                                 zaxis_title="Z mm", aspectmode="data"))
        st.plotly_chart(fig, width='stretch')
        st.caption("看点：点云应呈一张平滑起伏的面；离面飞点=误匹配或三角化失败。")
    else:
        # 单帧波剖面：PCA 面内坐标 + 逐片段去中位 + 沿实测传播向 ξ
        ck = None
        if os.path.exists(PINN_PT):
            import torch
            ck = torch.load(PINN_PT, map_location="cpu", weights_only=False)
        pts_frame_segs = [t[t[:, 0] == frame0] for t in trajs if np.any(t[:, 0] == frame0)]
        pts_all = np.vstack([t[:, 1:4] for t in trajs if len(t)])
        c, nvec = fit_plane(pts_all)
        _, vt = plane_vt(pts_all)
        if ck is not None and "rot" in ck:
            n2 = np.asarray(ck["rot"])[0]
        else:
            n2 = vt[0][:2]
        xs, ys = [], []
        for s in pts_frame_segs:
            d = s[:, 1:4] - c
            eta = d @ nvec
            uv = np.c_[d @ vt[0], d @ vt[1]]
            xs += list(uv @ n2)
            ys += list(eta - np.median(eta))
        xs, ys = np.array(xs), np.array(ys)
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", name="数据"))
        if len(xs) > 10:
            best = None
            for Lw in np.linspace(1500, 4000, 60):
                A = np.column_stack([np.sin(2 * np.pi * xs / Lw),
                                     np.cos(2 * np.pi * xs / Lw), np.ones_like(xs)])
                coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
                res = ys - A @ coef
                r2 = 1 - (res ** 2).sum() / max(1e-9, ((ys - ys.mean()) ** 2).sum())
                if best is None or r2 > best[0]:
                    best = (r2, Lw, coef)
            r2, Lw, coef = best
            A_fit = float(np.hypot(coef[0], coef[1]))
            xx = np.linspace(xs.min(), xs.max(), 300)
            yy = coef[0] * np.sin(2 * np.pi * xx / Lw) + coef[1] * np.cos(2 * np.pi * xx / Lw) + coef[2]
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines",
                                     name=f"拟合 λ={Lw:.0f} A={A_fit:.0f} R²={r2:.2f}"))
            st.caption(f"正弦拟合: λ={Lw:.0f} mm（理论 2502）, A={A_fit:.1f} mm（理论 40）, R²={r2:.2f}")
        fig.update_layout(height=450, xaxis_title="ξ mm", yaxis_title="η mm")
        st.plotly_chart(fig, width='stretch')
    if mode != "全序列点云(抽稀)":
        frame_player_tick("pc", list(range(f_min, f_max + 1)), fps)


def tab_judge():
    st.header("⑥ 互谱相位裁判")
    st.markdown("互谱相位法从轨迹对自身测相速度 c 与传播方向，不依赖 PINN、不依赖浪高仪，"
                "是全链路的内生裁判。理论相速：**1976 mm/s @ 名义 0.79Hz**；本窗口实测主频 0.783Hz"
                "的线性深水色散为 **1993 mm/s**（当前生产基线 1992 与之吻合，残余 -0.05%）。")

    st.subheader("泡沫影响 A/B 对比（eval_tracks 同一杆秤）")
    st.table({
        "指标": ["3D 段数(≥30帧)", "总 3D 点数", "≥100帧片段", "主峰率", "有效测速对",
               "c (mm/s)", "95% CI", "PINN 留出 R²"],
        "v1 旧生产(严格检测)": ["190", "10747", "21", "71%", "88", "2033 (+2.9%)", "[2000,2070]", "0.456"],
        "v3 含泡沫(对照组)": ["1474", "335819", "421", "93%", "97471", "2007 (+1.6%)", "[2006,2008]", "0.854"],
        "v3nf 泡沫过滤+带内分类(生产)": ["975", "292039", "369", "94%", "70135", "1992 (-0.05%)", "[1982,2002]", "0.964"],
        "DINO门控(对照,多对一)": ["2578", "472396", "841", "88%", "287092", "1997 (+1.1%)", "[1996,1997]", "—"],
    })
    st.caption("判据核验：连通域≥1500px² 拒天然水沫团（精度100%）；亮度判据已被数据推翻，未使用。"
               "生产行已按 8/4 口径修正（c=1992，片段级诚实 CI [1982,2002]）；除生产行外其余比较行"
               "为口径修正前的旧值，仅作相对趋势参考。下方按钮用已修复的 eval_tracks 实跑为准。")

    ds = st.radio("选择裁判对象", ["生产基线 v3nf_hung(泡沫过滤)", "对照组 v3app_hung(含泡沫)"],
                  horizontal=True, key="judge_ds")
    pkl = TRAJ3D if "生产" in ds else os.path.join(ROOT, "data/trajectories/trajectories_3d_v3app_hung_dino.pkl")
    if st.button("运行 eval_tracks（约 1 分钟，含 bootstrap）"):
        if not os.path.exists(pkl):
            st.warning("该数据集 3D 轨迹尚未生成（过滤链可能还在跑）")
        else:
            with st.spinner("评估中..."):
                r = subprocess.run([sys.executable, os.path.join(ROOT, "wave_modeling/eval_tracks.py"),
                                    pkl, ds], capture_output=True, text=True, cwd=ROOT,
                                   env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                st.session_state["eval_out"] = r.stdout + ("\n[stderr]\n" + r.stderr if r.returncode else "")
    if "eval_out" in st.session_state:
        st.code(st.session_state["eval_out"])

    for img_name, cap in [("diag_hovmoller.png", "Hovmöller 分箱占用 + PINN 场对比（应见对角传播条纹）"),
                          ("diag_xcorr.png", "互谱相位法测 c（两遍及 bootstrap CI）")]:
        p = os.path.join(REAL_RUN, img_name)
        if os.path.exists(p):
            st.image(p, caption=cap, width='stretch')
        else:
            st.info(f"{img_name} 不存在，运行: .venv_fs/Scripts/python.exe wave_modeling/diag_hovmoller_xcorr.py")


def tab_pinn():
    st.header("⑦ PINN 场重建")
    for img_name, cap in [("final_result.png", "数据 vs 重建（快照 + 稠密 Hovmöller + RMS 振幅 + 代表点时程）"),
                          ("verify_results.png", "一眼核验（η(t) + FFT 主峰 + 重合度）"),
                          ("field_comparison.png", "场对比（当前生产 pinn_real.pt）"),
                          ("field_comparison_c1992.png", "场对比（--c 1992 重训验证版，仅存档）")]:
        p = os.path.join(REAL_RUN, img_name)
        if os.path.exists(p):
            mt = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
            st.image(p, caption=f"{cap}  [生成 {mt}]", width='stretch')

    if os.path.exists(FIELD_NPZ):
        st.subheader("波面场交互浏览")
        import plotly.graph_objects as go
        f = load_field()
        xi, zeta, tt = f["xi"], f["zeta"], f["t"]
        comp = st.radio("分量", ["波动成分 eta_wave", "全量 eta", "覆盖率 coverage"], horizontal=True)
        if comp == "覆盖率 coverage":
            fig = go.Figure(go.Heatmap(x=zeta, y=xi, z=f["coverage"], colorscale="Viridis"))
            fig.update_layout(height=450, xaxis_title="ζ mm", yaxis_title="ξ mm")
            st.plotly_chart(fig, width='stretch')
        else:
            arr = f["eta_wave"] if comp.startswith("波动") else f["eta"]
            ti = st.slider("时间步", 0, arr.shape[2] - 1, 0)
            fig = go.Figure(go.Heatmap(x=zeta, y=xi, z=arr[:, :, ti], colorscale="RdBu_r",
                                       zmid=0, colorbar=dict(title="η mm")))
            t_lab = float(tt[ti]) if ti < len(tt) else ti
            fig.update_layout(height=450, title=f"η(ξ,ζ) @ t[{ti}]={t_lab:.2f}",
                              xaxis_title="ζ mm", yaxis_title="ξ mm")
            st.plotly_chart(fig, width='stretch')
            st.subheader("Hovmöller η(ξ,t)")
            zi = st.slider("ζ 切片", 0, arr.shape[1] - 1, arr.shape[1] // 2)
            fig2 = go.Figure(go.Heatmap(x=tt, y=xi, z=arr[:, zi, :], colorscale="RdBu_r",
                                        zmid=0, colorbar=dict(title="η mm")))
            fig2.update_layout(height=400, xaxis_title="t", yaxis_title="ξ mm")
            st.plotly_chart(fig2, width='stretch')
            st.caption("应看到清晰的对角条纹（波沿 ξ 传播）；条纹斜率 = 相速度。")

    logs = sorted([p for p in os.listdir(REAL_RUN) if p.endswith(".log")],
                  key=lambda p: os.path.getmtime(os.path.join(REAL_RUN, p)), reverse=True)
    if logs:
        with st.expander(f"训练日志 {logs[0]}"):
            st.text(open(os.path.join(REAL_RUN, logs[0]), encoding="utf-8", errors="replace").read())


def tab_reference():
    st.header("⑧ 基准对比（严志勇 2021 / 浪高仪）")
    st.markdown("""浪高仪数据来自论文数字化（`data/reference/yan2021_gauge/`）。
**注意**：浪高仪当时在拍摄区域外且与相机不同步——只能比统计量，相位对齐靠互相关反演，不代表物理同步。""")
    yan = load_yan()
    if not yan:
        st.warning("未找到 yan2021 数据文件")
        return
    if "gauge410" in yan:
        import plotly.graph_objects as go
        d = yan["gauge410"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d[:, 0], y=d[:, 1], mode="lines", name="浪高仪(严 图4-10)"))
        # 己方：最长 3D 轨迹段的 η(t)，去均值后互相关对齐
        try:
            trajs = load_traj3d()
            pts_all = np.vstack([t[:, 1:4] for t in trajs if len(t)])
            c, nvec = fit_plane(pts_all)
            segs = sorted(((len(t), t) for t in trajs if len(t) > 100), key=lambda s: -s[0])
            if segs:
                t0 = segs[0][1]
                fr = t0[:, 0].astype(int)
                eta = (t0[:, 1:4] - c) @ nvec
                eta = eta - np.median(eta)
                t_sec = fr / FPS
                t_sec = t_sec - t_sec[0]
                # 互相关对齐到浪高仪曲线前 4s
                g = d[:, 1] - d[:, 1].mean()
                e = np.interp(d[:, 0], t_sec, eta)
                e = e - e.mean()
                cc = np.correlate(g, e, "full")
                lag = np.argmax(cc) - (len(e) - 1)
                fig.add_trace(go.Scatter(x=d[:, 0], y=np.roll(e, lag) + d[:, 1].mean(),
                                         mode="lines", name="本项目最长轨迹 η(t)（互相关对齐）"))
                dt_ms = float(np.median(np.diff(d[:, 0]))) * 1000  # 从数据求采样间隔，勿硬编码
                st.caption(f"互相关对齐时延 {lag * dt_ms:.0f} ms；浪高仪曲线的绝对相位无物理意义，仅比波形/波幅。")
        except Exception as ex:
            st.info(f"己方轨迹叠加失败: {ex}")
        fig.update_layout(height=400, xaxis_title="t (s)", yaxis_title="η (mm)")
        st.plotly_chart(fig, width='stretch')

    st.subheader("统计量基准表")
    st.table({
        "量": ["波幅 A (mm)", "频率 f (Hz)", "波数 k (rad/m)", "波长 λ (m)"],
        "造波理论值": ["40.0", "0.79", "2.51157", "2.5017"],
        "浪高仪 50s(严)": ["40.09988", "0.78994", "2.51119", "-"],
        "严志勇双目 2959 帧": ["40.45338 ± 0.53", "0.782175", "2.46206 ± 0.03", "-"],
        "本项目目标": ["≈40（当前 PINN 中心 39.0）", "0.79", "-", "≈2.5"],
    })
    st.caption("来源：yan2021_reference_values.json。误差基准（严 vs 浪高仪）：波幅 0.88% / 波数 1.96% / 频率 0.98%。")


# ---------------------------------------------------------------- 主入口
def main():
    st.title("🌊 波浪重建全链路工序调试台")
    tabs = st.tabs(["总览", "①预处理/矫正", "②检测", "③2D跟踪", "④跨相机匹配",
                    "⑤3D点云", "⑥互谱裁判", "⑦PINN重建", "⑧基准对比"])
    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_preprocess()
    with tabs[2]:
        tab_detection()
    with tabs[3]:
        tab_tracks2d()
    with tabs[4]:
        tab_matching()
    with tabs[5]:
        tab_pointcloud()
    with tabs[6]:
        tab_judge()
    with tabs[7]:
        tab_pinn()
    with tabs[8]:
        tab_reference()

    st.sidebar.markdown("### 缓存")
    if st.sidebar.button("清空数据缓存（改数据后点）"):
        st.cache_data.clear()
        st.sidebar.success("已清空")
    st.sidebar.markdown("""### 说明
- 帧号一律 0 基（图像文件 = 帧号+1）
- 大数据（轨迹/场）首次加载需几秒，之后走缓存
- 改完任何一环，先跑 `run_full_pipeline.py` 刷新产物，再回这里逐页检查""")


if __name__ == "__main__":
    main()
