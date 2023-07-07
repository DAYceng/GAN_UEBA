import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

## test 2D GEN
# l0 = nn.Sequential(nn.Linear(7, 10368))
#
# layer1 = nn.BatchNorm2d(128)
# layer2 = nn.Upsample(scale_factor=2)
# layer3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
# layer4 = nn.BatchNorm2d(128, 0.8)
# layer5 = nn.LeakyReLU(0.2, inplace=True)
# layer6 = nn.Upsample(scale_factor=2)
# layer7 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
# layer8 = nn.BatchNorm2d(64, 0.8)
# layer9 = nn.LeakyReLU(0.2, inplace=True)
# layer10 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
# layer11 = nn.Linear(1296, 36)
# # layer11 = nn.Tanh()
#
# inputtensor0 = torch.rand(64, 7)
# inputtensor = torch.rand(64, 128, 8, 8)
# print(inputtensor0.shape)
#
#
# out0 = l0.forward(inputtensor0)
# out = out0.view(out0.shape[0], 128, 9, 9)
# out = layer1.forward(out)
# out = layer2.forward(out)
# out = layer3.forward(out)
# out = layer4.forward(out)
# out = layer5.forward(out)
# out = layer6.forward(out)
# out = layer7.forward(out)
# out = layer8.forward(out)
# out = layer9.forward(out)
# out = layer10.forward(out)
# out = out.view(out.shape[0], -1)
# out = layer11.forward(out)
#
# print(out.shape)
# # print(128 * ds_size ** 2)

## test 1D
# inputs = torch.rand([64, 36])
#
# padding = 1
# kernel_size = 3
# stride = 1
# C_out = 1
#
#
# x0 = torch.nn.Linear(36, 128 * 9 ** 2)
# x1 = torch.nn.Upsample(scale_factor=2)
# x2 = torch.nn.Conv1d(128, C_out, kernel_size, stride=stride, padding=padding)
# # x2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
#
# y = x0(inputs)
# y_ = y.view(y.shape[0], 128, 9, 9)
# y_ = x1(y_)
# # y_ = x2(y_)
# # y = x1(y)
# print(y_)
# print(y_.shape)

## test 2D DIS
inputs = torch.rand([64, 36])

init_size = 36 // 4  # 整数除法，下取整
l1 = nn.Sequential(nn.Linear(36, 128 * init_size ** 2))
# l0 = nn.Sequential(nn.Linear(36, 10368))
# layer1 = nn.Conv1d(6, 64, kernel_size=6, stride=2, padding=1)
layer1 = nn.Conv2d(1, 3, 3, 2, 1)
layer2 = nn.LeakyReLU(0.2, inplace=True)
layer3 = nn.Conv2d(3, 6, 3, 2, 1)
layer4 = nn.LeakyReLU(0.2, inplace=True)
layer5 = nn.Conv2d(6, 12, 3, 2, 1)
layer6 = nn.LeakyReLU(0.2, inplace=True)
# layer7 = nn.Conv2d(12, 24, 3, 2, 1)
# layer8 = nn.LeakyReLU(0.2, inplace=True)
ds_size = 36 // 12 * 4
adv_layer = nn.Sequential(nn.Linear(ds_size, 1), nn.Sigmoid())

# layer7 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
# layer8 = nn.BatchNorm2d(64, 0.8)
# layer9 = nn.LeakyReLU(0.2, inplace=True)
# layer10 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
# layer11 = nn.Linear(1296, 36)

# input_ = l1.forward(inputs)
# out0 = input_.view(input_.shape[0], 1, 36)
out0 = inputs.view(inputs.shape[0], 1, 6, 6)

out = layer1.forward(out0)
out = layer2.forward(out)  # torch.Size([64, 3, 3, 3])
out = layer3.forward(out)
out = layer4.forward(out)  # torch.Size([64, 6, 2, 2])
out = layer5.forward(out)
out = layer6.forward(out)  # torch.Size([64, 12, 1, 1])
# out = layer7.forward(out)
# out = layer8.forward(out)
out = out.view(out.shape[0], -1)

validity = adv_layer(out)  # torch.Size([64, 1])

print(validity.shape)
print(out0.shape)