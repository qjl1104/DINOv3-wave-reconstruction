import numpy as np, cv2, glob
calib = np.load('D:\\Research\\wave_reconstruction_project\\camera_calibration\\params\\stereo_calib_params_from_matlab_full.npz')
K1, D1, K2, D2 = calib['K_left'], calib['D_left'], calib['K_right'], calib['D_right']
R_mat = calib['R']
T_mat = calib['T']
img_sz = (2560, 1600)

lf = sorted(glob.glob(r'D:\Research\wave_reconstruction_project\data\left_images\*.*'))[0]
rf = lf.replace('left_images','right_images').replace('left','right')
imgL, imgR = cv2.imread(lf, 0), cv2.imread(rf, 0)

def test_rect(R, T, name):
    R1, R2, P1, P2, Q, r1, r2 = cv2.stereoRectify(K1, D1, K2, D2, img_sz, R, T, alpha=1)
    m1l, m2l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_sz, cv2.CV_32FC1)
    m1r, m2r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_sz, cv2.CV_32FC1)
    rectL = cv2.remap(imgL, m1l, m2l, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, m1r, m2r, cv2.INTER_LINEAR)
    
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(rectL, None)
    k2, d2 = sift.detectAndCompute(rectR, None)
    bf = cv2.BFMatcher()
    ms = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in ms if m.distance < 0.7 * n.distance]
    if len(good) == 0:
        print(f'{name}: 0 matches')
        return
    yd = np.array([abs(k1[m.queryIdx].pt[1] - k2[m.trainIdx].pt[1]) for m in good])
    print(f'{name}: matches={len(good)}, mean_y_err={yd.mean():.2f}px, med_y_err={np.median(yd):.2f}px')

print("Testing different conventions...")
test_rect(R_mat, T_mat, "Original")
test_rect(R_mat.T, T_mat, "R transposed")
test_rect(R_mat, -T_mat, "-T")
test_rect(R_mat.T, -T_mat, "R transposed and -T")
test_rect(R_mat.T, -R_mat.T @ T_mat, "Inverse Transform")
