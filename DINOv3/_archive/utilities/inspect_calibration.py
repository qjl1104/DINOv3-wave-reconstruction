import numpy as np
import sys

# --- !!! 修改这里的路径 !!! ---
CALIBRATION_FILE_PATH = r"D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_from_matlab_full.npz"

print(f"--- 正在检查标定文件: {CALIBRATION_FILE_PATH} ---")

try:
    calib_data = np.load(CALIBRATION_FILE_PATH)
except Exception as e:
    print(f"[FATAL ERROR] 无法加载文件: {e}")
    sys.exit(1)

print("\n文件中包含的所有键 (Keys):")
print(list(calib_data.keys()))

if 'Q' in calib_data:
    Q = calib_data['Q']
    print("\n--- 'Q' 矩阵 (4x4 重建矩阵) ---")
    print(Q)
    print("---------------------------------")
else:
    print("\n[错误] 文件中未找到 'Q' 矩阵！")

print("\n--- 检查完毕 ---")