import cv2
import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import scipy.io as sio
import matplotlib.pyplot as plt
import xlrd
from PIL import Image
import os

path = r'D:\安装包\外协\某水面船测距文件'
rpath = r'D:\安装包\外协\某水面船测距文件'
rrpath = r'D:\安装包\外协\其它测距文件\测距数据已修正'


class Nstr:
    def __init__(self, arg):
        self.x = arg

    def __sub__(self, other):
        c = self.x.replace(other.x, "")
        return c


def ymaxminnorm(mat, num):
    maxcols = []
    mincols = []
    tt = []
    for i in range(1, num + 1):
        array = mat[:, i - 1]
        max = array.max()
        min = array.min()
        data_shape = len(array)
        maxcols.append(max)
        mincols.append(min)
        t = np.empty(data_shape)
        for j in range(data_shape):
            t[j] = data_shape * array[j] / sum(array)
        tt.append(t)
    tt = np.array(tt)
    tt = tt.reshape(data_shape, num)
    return tt, maxcols, mincols


def maxminnorm(mat, num):
    maxcols = []
    mincols = []
    tt = []
    for i in range(1, num + 1):
        array = mat[i - 1, :]
        max = array.max()
        min = array.min()
        data_shape = len(array)
        maxcols.append(max)
        mincols.append(min)
        t = np.empty(data_shape)
        for j in range(data_shape):
            t[j] = (array[j] - min) / (max - min)
        tt.append(t)
    tt = np.array(tt)
    tt = tt.reshape(num, data_shape)
    return tt, maxcols, mincols


def readall(array, num, inn):
    yy = []
    inn = str(inn)
    print(inn)
    ipp = Nstr(inn) - Nstr("RC-")
    ipp = Nstr(ipp) - Nstr(".xls")
    for i in range(1, num + 1):
        sheetName = str(ipp) + '-R' + str(i) + '#'
        list_ = 'list_' + str(i)
        list0 = array[sheetName]
        r = list0.nrows
        c = list0.ncols
        x = np.array(list0.col_values(0, 1))
        y = np.array(list0.col_values(1, 1))
        yy.append(y)

    yy = np.array(yy)
    yy = yy.reshape(num, r - 1)
    return r, c, x, yy


i, j = 0, 0
roor = r'D:\安装包\外协\某水面船测距文件\测距数据待修正'
for dirpath, dirnames, filenames in os.walk(roor):
    for filepath in filenames:
        i += 1
        wb = xlrd.open_workbook(filename=roor + "/" + filepath)

        row, col, wx, wy = readall(wb, 8, filepath)

        b_data, maxy, miny = ymaxminnorm(np.array(wy), len(wy[0, :]))
        a_data, maxx, minx = maxminnorm(np.array(b_data), 8)
        img_processed = Image.fromarray(np.uint8(wy / 1000 * 255))

        print(img_processed.mode)
        Image.Image.save(img_processed, fp=path + '\Val' + '\Val' + str(i) + '.png')

root = r'D:\安装包\外协\某水面船测距文件\测距数据已修正'
for dirpath, dirnames, filenames in os.walk(root):
    for filepath in filenames:
        j += 1
        rwb = xlrd.open_workbook(filename=root + "/" + filepath)
        rrow, rcol, rwx, rwy = readall(rwb, 8, filepath)
        rimg_processed = Image.fromarray(np.uint8(rwy / 1000 * 255))
        Image.Image.save(rimg_processed, fp=rpath + '\Tra' + '\Tra' + str(j) + '.png')

root = r'D:\安装包\外协\其它测距文件\测距数据已修正'
for dirpath, dirnames, filenames in os.walk(root):
    for filepath in filenames:
        j += 1
        rwb = xlrd.open_workbook(filename=root + "/" + filepath)
        rrow, rcol, rwx, rwy = readall(rwb, 8, filepath)
        rrimg_processed = Image.fromarray(np.uint8(rwy / 1000 * 255))
        Image.Image.save(rrimg_processed, fp=rpath + '\Val2' + '\Val' + str(j) + '.png')
'''
n_blocks = 5  # 生成器中的残差块数量
n_epochs = 100  # 训练迭代次数
batch_size = 64  # 批量大小
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择在GPU还是CPU上训练


sheetName1 = "run102-R1#"
sheetName2 = "run102-R2#"
sheetName3 = "run102-R3#"
sheetName4 = "run102-R4#"
sheetName5 = "run102-R5#"
sheetName6 = "run102-R6#"
sheetName7 = "run102-R7#"
sheetName8 = "run102-R8#"
w1 = wb[sheetName1]
w2 = wb[sheetName2]
w3 = wb[sheetName3]
w4 = wb[sheetName4]
w5 = wb[sheetName5]
w6 = wb[sheetName6]
w7 = wb[sheetName7]
w8 = wb[sheetName8]

row = w1.nrows
col = w1.ncols
w1x = w1.col_values(0, 1)
w1y = w1.col_values(1, 1)
w2x = w2.col_values(0, 1)
w2y = w2.col_values(1, 1)
w3x = w3.col_values(0, 1)
w3y = w3.col_values(1, 1)
w4x = w4.col_values(0, 1)
w4y = w4.col_values(1, 1)
w5x = w5.col_values(0, 1)
w5y = w5.col_values(1, 1)
w6x = w6.col_values(0, 1)
w6y = w6.col_values(1, 1)
w7x = w7.col_values(0, 1)
w7y = w7.col_values(1, 1)
w8x = w8.col_values(0, 1)
w8y = w8.col_values(1, 1)
w1y, w2y, w3y, w4y, w5y, w6y, w7y, w8y = maxminnorm(w1y), maxminnorm(w2y), maxminnorm(w3y), maxminnorm(w4y), maxminnorm(
    w5y), maxminnorm(w6y), maxminnorm(w7y), maxminnorm(w8y)

x_data = data[1, :]
x_data, maxx, minx = maxminnorm(np.array(x_data))
ds_x = x_data[:, np.newaxis]

y_data = data[2, :]
y_data = y_data - np.mean(y_data)
y_data, maxy, miny = maxminnorm(np.array(y_data))
ds_y = y_data[:, np.newaxis]
'''
