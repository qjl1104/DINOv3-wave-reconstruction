import numpy as np
import os
import cv2
import matplotlib.pyplot as plt # 虽然这里没用到 matplotlib 显示，但保留以防万一

# --- 全局变量用于鼠标回调 ---
drawing = False # 如果正在按下鼠标，则为 True
ix, iy = -1, -1 # 起始坐标
rect_final = None # 最终选定的矩形 (x, y, w, h)
img_display = None # 用于绘制矩形的图像副本

# --- 鼠标回调函数 ---
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect_final, img_display
    img_copy = param['img_copy'] # 获取原始图像副本
    window_name = param['window_name']

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect_final = None # 清除旧矩形
        # 按下时重置显示图像，清除之前的红框
        cv2.imshow(window_name, img_copy)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 创建临时副本以显示当前拖拽的矩形
            img_temp = img_copy.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2) # 绿色拖拽框
            cv2.imshow(window_name, img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 最终绘制矩形 (在原始副本上绘制最终红框)
        img_final_display = img_copy.copy() # 创建新的副本显示最终结果
        cv2.rectangle(img_final_display, (ix, iy), (x, y), (0, 0, 255), 2) # 用红色确认最终矩形
        cv2.imshow(window_name, img_final_display)
        # 计算并存储最终矩形坐标 (确保 w 和 h 是正数)
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        rect_final = (x1, y1, x2 - x1, y2 - y1)
        print(f"\n选定的 ROI 区域:")
        print(f"  x = {rect_final[0]}")
        print(f"  y = {rect_final[1]}")
        print(f"  width = {rect_final[2]}")
        print(f"  height = {rect_final[3]}")
        print("\n按 's' 保存此 ROI 并退出, 按 'r' 重新绘制, 按 'q' 不保存退出。")

# --- 配置 ---
try:
    from sparse_reconstructor_1013_gemini_v2 import Config
    CALIBRATION_FILE = Config.CALIBRATION_FILE
    LEFT_IMAGE_DIR = Config.LEFT_IMAGE_DIR # 仍然需要知道图像文件夹
    RIGHT_IMAGE_DIR = Config.RIGHT_IMAGE_DIR
    print(f"从 Config 加载标定文件路径: {CALIBRATION_FILE}")
    print(f"从 Config 加载图像文件夹路径: {LEFT_IMAGE_DIR}, {RIGHT_IMAGE_DIR}")
except (ImportError, AttributeError):
    print("无法从 Config 加载路径，请确保下面的 CALIBRATION_FILE 和图像文件夹路径正确。")
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.dirname(PROJECT_ROOT)
    CALIBRATION_FILE = os.path.join(DATA_ROOT, "camera_calibration", "params", "stereo_calib_params_from_matlab_full.npz")
    LEFT_IMAGE_DIR = os.path.join(DATA_ROOT, "data", "left_images") # 仍然需要知道图像文件夹
    RIGHT_IMAGE_DIR = os.path.join(DATA_ROOT, "data", "right_images")
    print(f"使用手动指定的标定文件路径: {CALIBRATION_FILE}")
    print(f"使用手动指定的图像文件夹路径: {LEFT_IMAGE_DIR}, {RIGHT_IMAGE_DIR}")

# --- !! 选择用于可视化检查的 **右** 图像 !! ---
RIGHT_IMAGE_NAME = "right0001.bmp" # 例如
# --------------------------------------------

# --- 加载标定数据 ---
try:
    calib_data = np.load(CALIBRATION_FILE)
    print(f"\n成功加载标定文件: {CALIBRATION_FILE}")
except FileNotFoundError:
    print(f"\n错误: 找不到标定文件 '{CALIBRATION_FILE}'。请检查路径。")
    exit()
except Exception as e:
    print(f"\n错误: 加载标定文件时出错: {e}")
    exit()

# --- 提取 **右** 相机校正参数 ---
required_keys = ['map1_right', 'map2_right']
missing_keys = [key for key in required_keys if key not in calib_data.keys()]
if missing_keys:
    print(f"\n错误: 标定文件缺少以下必需的键: {missing_keys}")
    calib_data.close()
    exit()
else:
    map1_right = calib_data['map1_right']
    map2_right = calib_data['map2_right']
    original_roi_right = calib_data.get('roi_right', None) # 获取旧 ROI 以便比较

# --- 加载并校正 **右** 图像 ---
right_img_path = os.path.join(RIGHT_IMAGE_DIR, RIGHT_IMAGE_NAME)
if not os.path.exists(right_img_path):
    print(f"错误: 找不到右图像 '{right_img_path}'")
    calib_data.close()
    exit()

print(f"加载图像: {RIGHT_IMAGE_NAME}")
img_right_raw = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
if img_right_raw is None:
    print("错误: 无法加载选定的图像。")
    calib_data.close()
    exit()

# 应用校正
img_right_rect = cv2.remap(img_right_raw, map1_right, map2_right, cv2.INTER_LINEAR)
print("图像校正完成。")

# --- 交互式 ROI 选择 ---
window_name = 'Select ROI on Rectified **Right** Image' # 修改窗口标题
cv2.namedWindow(window_name)

# 创建图像副本用于绘制
img_display_initial = cv2.cvtColor(img_right_rect, cv2.COLOR_GRAY2BGR)

# 如果存在旧 ROI，先画出来作为参考 (用蓝色)
if original_roi_right is not None and len(original_roi_right) == 4:
    ox, oy, ow, oh = map(int, original_roi_right)
     # 检查边界，防止绘制到图像外导致错误
    h_img, w_img = img_display_initial.shape[:2]
    if ox + ow <= w_img and oy + oh <= h_img and ox >= 0 and oy >= 0:
        cv2.rectangle(img_display_initial, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)
        print(f"\n蓝色框显示的是 .npz 文件中原始的 ROI (Right): x={ox}, y={oy}, w={ow}, h={oh}")
    else:
        print(f"\n警告: .npz 文件中原始的 ROI (Right) 坐标无效或超出边界: x={ox}, y={oy}, w={ow}, h={oh}")


print("\n--- 请在图像窗口中操作 ---")
print("1. 按下鼠标左键并拖拽以绘制矩形框。")
print("2. 松开鼠标左键以确定矩形。")
print("3. 查看控制台输出的 ROI 坐标。")
print("4. 如果满意，按 's' 键保存并退出。")
print("5. 如果不满意，按 'r' 键清除矩形并重新绘制。")
print("6. 按 'q' 键直接退出而不保存。")


# 将图像副本传递给回调函数
img_clean_copy = img_display_initial.copy() # 保存一个没有红框的干净副本
param = {'img_copy': img_clean_copy, 'window_name': window_name}
cv2.setMouseCallback(window_name, draw_rectangle, param)

cv2.imshow(window_name, img_display_initial) # 显示初始图像

final_selected_roi = None

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # 保存
        if rect_final and rect_final[2] > 0 and rect_final[3] > 0:
            final_selected_roi = rect_final
            print("\nROI 已选定并保存。")
            break
        else:
            print("\n错误：请先绘制一个有效的矩形框再按 's'。")
    elif key == ord('r'): # 重置
        print("\n重置绘图区域，请重新绘制...")
        rect_final = None
        img_clean_copy = cv2.cvtColor(img_right_rect, cv2.COLOR_GRAY2BGR) # 重新加载无框图像
        if original_roi_right is not None and len(original_roi_right) == 4: # 重绘旧 ROI 参考
            ox, oy, ow, oh = map(int, original_roi_right)
            h_img, w_img = img_clean_copy.shape[:2]
            if ox + ow <= w_img and oy + oh <= h_img and ox >= 0 and oy >= 0:
                cv2.rectangle(img_clean_copy, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)
        param['img_copy'] = img_clean_copy.copy() # 更新回调参数中的副本
        cv2.imshow(window_name, img_clean_copy) # 显示无红框图像
    elif key == ord('q'): # 退出
        print("\n退出，未保存选定的 ROI。")
        break
    elif key == 27: # ESC 键也退出
        print("\n退出，未保存选定的 ROI。")
        break


cv2.destroyAllWindows()
calib_data.close()

# --- 输出最终结果 ---
if final_selected_roi:
    print("\n--- 最终选定的 **右相机** ROI 坐标 ---") # 修改提示
    print(f"  x = {final_selected_roi[0]}")
    print(f"  y = {final_selected_roi[1]}")
    print(f"  width = {final_selected_roi[2]}")
    print(f"  height = {final_selected_roi[3]}")
    print("\n请将这些值手动更新到你的训练脚本 (`sparse_reconstructor... .py`) 的 `RectifiedWaveStereoDataset` 类中，")
    print("覆盖从 .npz 文件加载的 `self.roi_right`。")
else:
    print("\n未选定最终 ROI。")

print("\n分析完成。")
