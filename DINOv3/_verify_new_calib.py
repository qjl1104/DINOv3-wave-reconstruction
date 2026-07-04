import numpy as np
import cv2
import os
import glob

# Load the NEW calibration (we just ran it with alpha=0)
calib_path = r'D:\Research\wave_reconstruction_project\DINOv3\paper_params_recalculated.npz'
if not os.path.exists(calib_path):
    print("Calib file not found!")
    exit()

calib = np.load(calib_path)
m1l, m2l = calib['map1_left'], calib['map2_left']
m1r, m2r = calib['map1_right'], calib['map2_right']
Q = calib['Q']

print(f'=== 新标定核验 (alpha=0) ===')
print(f'新焦距: {Q[2,3]:.1f}')
print(f'新基线: {1.0/Q[3,2]:.1f} mm')

# Load images
lf = sorted(glob.glob(r'D:\Research\wave_reconstruction_project\data\left_images\*.*'))
l = cv2.imread(lf[0], 0)
r = cv2.imread(lf[0].replace('left_images','right_images').replace('left','right'), 0)

# Rectify
lr = cv2.remap(l, m1l, m2l, cv2.INTER_LINEAR)
rr = cv2.remap(r, m1r, m2r, cv2.INTER_LINEAR)

# Run SIFT to check geometric accuracy
sift = cv2.SIFT_create()
k1, d1 = sift.detectAndCompute(lr, None)
k2, d2 = sift.detectAndCompute(rr, None)
bf = cv2.BFMatcher()
ms = bf.knnMatch(d1, d2, k=2)

good = [m for m, n in ms if m.distance < 0.7 * n.distance]
yd = np.array([abs(k1[m.queryIdx].pt[1] - k2[m.trainIdx].pt[1]) for m in good])
ds = np.array([k1[m.queryIdx].pt[0] - k2[m.trainIdx].pt[0] for m in good])

print(f'\nSIFT 匹配点对数量: {len(good)}')
print(f'极线误差 (Y方向差值): 平均={yd.mean():.2f} 像素, 中位数={np.median(yd):.2f} 像素')
print(f'极线误差 < 1像素 的比例: {(yd<1).mean()*100:.1f}%')
print(f'极线误差 < 3像素 的比例: {(yd<3).mean()*100:.1f}%')

pos = ds[ds > 0]
if len(pos) > 0:
    print(f'视差分布 (正值): 平均={pos.mean():.1f}, 范围=[{pos.min():.1f}, {pos.max():.1f}]')
