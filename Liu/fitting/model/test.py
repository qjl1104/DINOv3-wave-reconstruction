import argparse
import time

import cv2
import torchvision
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import TensorDataset

from Network import *
from dataset import *
from loss import *
from model.main import n_blocks

model = Generator(n_blocks)
model.load_state_dict(torch.load('Generator.pth', map_location=torch.device('cpu')))
model.eval()

test_transform = transforms.ToTensor()
testimg = Image.open('D:\安装包\外协\某水面船测距文件\Val/Val1.png')
tempImg = testimg.convert('RGB')
timg = np.array(tempImg) / 255
# timg = addGaussNoise(timg, 30)
timg = torch.tensor(timg.transpose(2, 0, 1)).float().unsqueeze(0)
dnimg = model(timg)[0, :, :, :]
dnimg = dnimg.detach().numpy().transpose((1, 2, 0))
# 噪声图像
timg = Image.fromarray(
    np.uint8(cv2.normalize(timg.squeeze().detach().numpy().transpose(1, 2, 0), None, 0, 255, cv2.NORM_MINMAX)))
timg.save('first.png')

# 降噪后的图像
img = Image.fromarray(np.uint8(cv2.normalize(dnimg, None, 0, 255, cv2.NORM_MINMAX)))
gray_img = img.convert('L')
gray_img.save('ans_test.png')
