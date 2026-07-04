import os
import datetime

# 定义我们要寻找的目标文件类型和关键词
TARGETS = {
    "1. 相机标定 (最高优先级)": {
        "keywords": ["calib", "matrix", "param", "intrinsics", "extrinsics", "left", "right", "xml", "yaml", "mat",
                     "json"],
        "weight": 10,
        "desc": "包含 K_left, K_right, R, T 等矩阵，没有这个没法做三维重建。"
    },
    "2. 特征提取与匹配 (核心算法)": {
        "keywords": ["feature", "match", "sift", "superpoint", "glue", "stereo", "disparity", "cost", "epipolar",
                     "triangulat"],
        "weight": 8,
        "desc": "论文2.2.4节的自编算子和SuperPoint/SuperGlue实现。"
    },
    "3. 深度学习模型 (波面预测)": {
        "keywords": ["train", "model", "net", "convlstm", "lstm", "gru", "predict", "wave", "loss", "ssim", "keras",
                     "torch"],
        "weight": 7,
        "desc": "论文第四章的ConvLSTM网络结构和训练代码。"
    },
    "4. 数据处理/主程序": {
        "keywords": ["main", "run", "process", "utils", "data", "loader", "pointcloud", "reconstruct"],
        "weight": 5,
        "desc": "整个流程的入口或数据预处理脚本。"
    }
}


def scan_directory(root_dir="."):
    output_lines = []

    header = f"扫描时间: {datetime.datetime.now()}\n正在扫描目录: {os.path.abspath(root_dir)}\n"
    print(header)
    output_lines.append(header)
    output_lines.append("-" * 60)
    output_lines.append(f"{'文件名':<40} | {'推测类型':<20} | {'匹配度'}")
    output_lines.append("-" * 60)

    found_files = {key: [] for key in TARGETS}
    file_count = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 跳过编译文件、系统文件和自身
            if file.endswith(('.pyc', '.git', '.idea', '__pycache__',
                              '.DS_Store')) or file == 'scan_for_thesis_files.py' or file == 'scan_result.txt':
                continue

            file_count += 1
            # 计算匹配分数
            file_lower = file.lower()
            path_lower = os.path.join(root, file).lower()

            best_match = None
            max_score = 0

            for category, info in TARGETS.items():
                score = 0
                for kw in info['keywords']:
                    if kw in file_lower:
                        score += 3  # 文件名匹配权重极高
                    elif kw in path_lower:
                        score += 1  # 路径匹配权重低

                if score > 0:
                    total_score = score * info['weight']
                    if total_score > max_score:
                        max_score = total_score
                        best_match = category

            if best_match:
                rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                found_files[best_match].append(rel_path)
                line = f"{rel_path[:40]:<40} | {best_match.split(' ')[1]:<20} | {'★' * min(5, max_score // 5)}"
                print(line)
                output_lines.append(line)

    output_lines.append("-" * 60)
    summary_header = "\n扫描总结 (请将以下内容发给AI):"
    print(summary_header)
    output_lines.append(summary_header)

    has_critical = False
    for category, files in found_files.items():
        if files:
            cat_header = f"\n【{category}】"
            print(cat_header)
            output_lines.append(cat_header)

            desc = f"  说明: {TARGETS[category]['desc']}"
            print(desc)
            output_lines.append(desc)

            for f in files:
                file_line = f"  - {f}"
                print(file_line)
                output_lines.append(file_line)

            if "相机标定" in category:
                has_critical = True

    if not has_critical:
        warn = "\n⚠️ 警告：未找到明显的相机标定文件！请手动查找 .mat 或 .xml 文件。"
        print(warn)
        output_lines.append(warn)

    # 将结果写入文件
    with open('scan_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n✅ 扫描完成！共扫描 {file_count} 个文件。")
    print(f"📄 结果已保存至当前目录下的 'scan_result.txt'。")
    print(f"👉 请打开 'scan_result.txt'，全选复制内容，并发给AI。")


if __name__ == "__main__":
    scan_directory()