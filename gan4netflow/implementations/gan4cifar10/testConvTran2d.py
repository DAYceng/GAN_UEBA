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

latent_size = 64
n_channel = 3
n_g_feature = 64
inputtensor = torch.rand(64, 64, 1, 1)

layer = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
    nn.BatchNorm2d(4 * n_g_feature)
)

print(inputtensor.shape)
out = layer(inputtensor)
print(out.shape)

