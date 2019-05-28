# 深層学習ライブラリpytorch
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 可視化モジュール
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import time


class CNN(nn.Module):
    def __init__(self, image_size, num_classes, C=3):
        super(CNN, self).__init__()
        self.shallow = nn.Sequential(
            nn.Conv2d(C, 2*C, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(2*C),
            nn.ReLU(),
            nn.Conv2d(2*C, 4*C, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(4*C),
            nn.ReLU(),
            nn.ConvTranspose2d(4*C, 8*C, kernel_size=4, padding=0, stride=4),
            nn.Sigmoid()
        )
        self.GAP = nn.AvgPool2d(image_size)
        self.linear = nn.Linear(8*C, num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        out = self.shallow(x)
        out = self.GAP(out)
        out = torch.squeeze(out)
        out = self.linear(out)
        out = self.softmax(out)
        return out
    
    def get_cam(self, x, idx):
        self.eval()
        camout = self.shallow(x)
        # act_mapはGAP前の特徴マップ, (C x image_size x image_size)
        act_map = camout[0]
        C, H, W = act_map.size()
        # weightsは全結合層における重みの値, (C x num_classes)
        weights = self.linear.weight.data
        N, _ = weights.size()
        # camの計算 (num_classes x image_size x image_size)
        cam = torch.mm(weights, act_map.view(C, H*W)).view(N, H, W)
        # 特定のクラスの重みのみ取り出す
        cam = cam[idx]
        maxval = cam.max()
        minval = cam.min()
        cam = (cam - minval) / (maxval - minval)
        cam = cam.cpu().detach().numpy()
        return cam

class ResidualBlock(nn.Module):
    def __init__(self, C):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(C),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        return x + self.model(x)

    
class CNN_res(nn.Module):
    def __init__(self, image_size, num_classes, residual_num=5, C=3):
        super(CNN_res, self).__init__()
        self.shallow = []
        for i in range(residual_num):
            self.shallow.append(ResidualBlock(C))
        self.shallow = nn.Sequential(*self.shallow)
        self.GAP = nn.AvgPool2d(image_size)
        self.linear = nn.Linear(C, num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        out = self.shallow(x)
        out = self.GAP(out)
        out = torch.squeeze(out)
        out = self.linear(out)
        out = self.softmax(out)
        return out
    
    def get_cam(self, x, idx):
        self.eval()
        camout = self.shallow(x)
        # act_mapはGAP前の特徴マップ, (C x image_size x image_size)
        act_map = camout[0]
        C, H, W = act_map.size()
        # weightsは全結合層における重みの値, (C x num_classes)
        weights = self.linear.weight.data
        N, _ = weights.size()
        # camの計算 (num_classes x image_size x image_size)
        cam = torch.mm(weights, act_map.view(C, H*W)).view(N, H, W)
        # 特定のクラスの重みのみ取り出す
        cam = cam[idx]
        maxval = cam.max()
        minval = cam.min()
        cam = (cam - minval) / (maxval - minval)
        cam = cam.cpu().detach().numpy()
        return cam

# ミニバッチ化を行う関数
def collater(sample):
    images = []
    classes = []
    for i in sample:
        images.append(i[0])
        classes.append(i[1])
    im = torch.stack(images)
    cl = torch.tensor(classes)
    return im, cl

def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, std=INITSTD)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

# camを画像に乗せてヒートマップにする関数．画像とcamを入力する
def cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	return cam