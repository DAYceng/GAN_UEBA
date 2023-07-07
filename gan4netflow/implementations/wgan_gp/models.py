import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        if opt.dataset == 'mnist':
            self.model = nn.Sequential(
                *block(opt.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

        elif opt.dataset == 'netflow':
            # self.netflowmodel = nn.Sequential(
            #     nn.Conv1d(),
            #     nn.BatchNorm1d(),
            #     nn.ReLU(),
            #
            #     nn.Tanh()
            # )
            self.netflowmodel = nn.Sequential(
                # *block(opt.latent_dim4flow, 36, normalize=False),
                # *block(36, 72),
                # *block(72, 144),
                # *block(144, 288),
                # nn.Linear(288, int(np.prod(netmatrix))),
                # nn.Tanh()
                * block(opt.latent_dim4flow, 14, normalize=False),
                *block(14, 28),
                *block(28, 56),
                *block(56, 128),
                nn.Linear(128, int(np.prod(netmatrix))),
                nn.Tanh()
            )

        elif opt.dataset == 'cifar10':
            #转置卷积
            self.latent_size = 64
            self.n_channel = 3
            self.n_g_feature = 64
            # self.main =
            self.main = nn.Sequential(
                nn.ConvTranspose2d(self.latent_size, 4 * self.n_g_feature, kernel_size=4, bias=False),
                nn.BatchNorm2d(4 * self.n_g_feature),
                nn.ReLU(),  # (32-1)x1+2*0+(4-1)+1=35

                nn.ConvTranspose2d(4 * self.n_g_feature, 2 * self.n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2 * self.n_g_feature),
                nn.ReLU(),  # (35-1)x2+2*1+(4-1)+1=74

                nn.ConvTranspose2d(2 * self.n_g_feature, self.n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.n_g_feature),
                nn.ReLU(),  # (74-1)x2+2*1+(4-1)+1=152

                nn.ConvTranspose2d(self.n_g_feature, self.n_channel, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()  # (152-1)x2+2*1+(4-1)+1=308
            )


    def forward(self, z):
        if opt.dataset == 'mnist':
            img = self.model(z)
            output_img = img.view(img.shape[0], *img_shape)
            return output_img

        elif opt.dataset == 'netflow':
            flow = self.netflowmodel(z)

            return flow

        elif opt.dataset == 'cifar10':
            # # 全连接
            # img = self.model(z)
            # output_img = img.view(img.shape[0], *img_shape)

            # #卷积
            # output_main = self.main(z)  # torch.Size([100, 3, 32, 32])
            # output_img = F.sigmoid(output_main + self.output_bias)  # torch.Size([100, 3, 32, 32])
            # # output_img = F.sigmoid(output_main) #不加偏置

            # #转置卷积
            output = self.main(z)
            output_img = output.detach()
            return output_img

        # return output_img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        if opt.dataset == 'mnist':
            self.model1 = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
            )
            self.soft_output = nn.Sigmoid()

        elif opt.dataset == 'netflow':
            self.flowdiscr = nn.Sequential(
                nn.Linear(int(np.prod(netmatrix)), 24),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(24, 12),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(12, 1),
            )
            self.soft_output = nn.Sigmoid()

            # self.flowdiscr = nn.Sequential(
            #     nn.Linear(int(np.prod(netmatrix)), 7),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.Linear(7, 1),
            # )
            # self.soft_output = nn.Sigmoid()

        elif opt.dataset == 'cifar10':
            # # 使用全连接层
            # self.model1 = nn.Sequential(
            #     nn.Linear(int(np.prod(img_shape)), 1536),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.Linear(1536, 768),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.Linear(768, 512),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.Linear(512, 256),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.Linear(256, 1),
            # )
            # #使用卷积层
            self.latent_size = opt.latent_dim  # 256
            self.dropout = opt.dropout  # 0.2
            self.output_size = opt.output_size  # 1

            self.model2 = nn.Sequential(
                # 输入形状：torch.Size([64, 3, 32, 32])
                # 计算方法：
                # ([(输入大小)+2*padding-(卷积核-1)]/步长)+1
                # ([(32)+2*0-(5-1)]/1)+1 = 28
                # torch.Size([64, 32, 28, 28])
                nn.Conv2d(3, 32, 5, stride=1, bias=True),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=self.dropout),

                # ([(28)+2*0-(4-1)]/2)+1 = 13（下取整）
                # torch.Size([64, 64, 13, 13])
                nn.Conv2d(32, 64, 4, stride=2, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=self.dropout),

                # ([(13)+2*0-(4-1)]/1)+1 = 11（下取整）公式计算得11，程序计算得10
                # torch.Size([64, 128, 10, 10])
                nn.Conv2d(64, 128, 4, stride=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=self.dropout),

                nn.Conv2d(128, 256, 4, stride=2, bias=False),# torch.Size([64, 256, 4, 4])
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=self.dropout),

                nn.Conv2d(256, 512, 4, stride=1, bias=False),# torch.Size([64, 512, 1, 1])
                nn.BatchNorm2d(512),
                nn.LeakyReLU(inplace=False),
                nn.Dropout2d(p=self.dropout),
            )

            #
            ds_size = opt.img_size // 2 ** 4 #双星号为乘方
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())


    def forward(self, img):
        if opt.dataset == 'mnist':
            img_flat = img.view(img.shape[0], -1)# torch.Size([64, 784]),28*28*1
            validity = self.model1(img_flat)# torch.Size([64, 1])
            softout = self.soft_output(validity) # 获取概率

        if opt.dataset == 'netflow':
            # img_flat = img.view(img.shape[0], -1)  # torch.Size([64, 784]),28*28*1
            validity = self.flowdiscr(img)  # torch.Size([64, 1])
            softout = self.soft_output(validity)  # 获取概率


        if opt.dataset == 'cifar10':
            # img_flat = img.view(img.shape[0], -1)  # torch.Size([64, 784]),28*28*1
            # validity = self.model1(img_flat)  # torch.Size([64, 1])

            output_img = self.model2(img)  # torch.Size([100, 512, 1, 1])
            # output_final = self.final(output_img)  # torch.Size([100, 1, 1, 1])
            out = output_img.view(output_img.shape[0], -1)
            if self.output_size == 1:
                # validity = F.sigmoid(output_final)  # torch.Size([100, 1, 1, 1])
                validity = self.adv_layer(out)
                softout = validity

            # output_img = self.model2(img)
            # output = output_img.view(output_img.shape[0], -1)
            # validity = self.adv_layer(output)

        return validity, softout
