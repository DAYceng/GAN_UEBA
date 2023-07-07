import torch
import torch.nn as nn
import numpy as np
from config import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 50)
        )
    # input: torch.Size([128, 784])
    def forward(self, X):
        encode = self.layers(X)  # torch.Size([128, 50])
        return encode

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(50 + 10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 28 * 28),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Sigmoid()  # nn.Tanh()
        )

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)  # torch.Size([128, 60])
        img = self.layers(zc)  # torch.Size([128, 784]),28*28*1
        output_img = img.view(img.shape[0], *img_shape)  # torch.Size([128, 1, 28, 28])
        return output_img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + 50 + 10, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    # input: torch.Size([128, 784])
    def forward(self, img, z, c):
        img_flat = img.view(img.shape[0], -1)    # torch.Size([128, 784])
        Xzc = torch.cat([img_flat, z, c], dim=1)  # torch.Size([128, 844])
        validity = self.layers(Xzc)  # torch.Size([128, 1])
        return validity