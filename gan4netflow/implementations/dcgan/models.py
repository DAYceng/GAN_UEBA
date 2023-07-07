from torch import nn
from config import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        if opt.dataset == 'mnist':
            self.init_size = opt.img_size // 4  # 整数除法，下取整
            self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

            # input torch.Size([64, 128, 8, 8])
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),  # torch.Size([64, 128, 16, 16])
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),  # torch.Size([64, 128, 32, 32])
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),  # torch.Size([64, 1, 32, 32])
                nn.Tanh(),
            )
        elif opt.dataset == 'netflow':
            self.init_size = opt.flow_size // 4  # 整数除法，下取整
            self.l1 = nn.Sequential(nn.Linear(opt.latent_dim4flow, 128 * self.init_size ** 2))

            # input torch.Size([64, 128, 9, 9])
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),  # 64, 128, 18, 18
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),  # 64, 64, 36, 36
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),  # 64, 1, 36, 36
                nn.Tanh(),
            )

            # Downsample the flattened data
            # input torch.Size([64, 1296])
            self.out_layer = nn.Sequential(nn.Linear((4 * self.init_size) ** 2, 36))

    # z:torch.Size([64, 100])
    def forward(self, z):
        if opt.dataset == 'mnist':
            out = self.l1(z)  # torch.Size([64, 8192])
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)  # torch.Size([64, 128, 8, 8])
            img = self.conv_blocks(out)  # torch.Size([64, 1, 32, 32])
            return img
        elif opt.dataset == 'netflow':
            #2D
            out = self.l1(z)  # torch.Size([64, 10368])
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)  # torch.Size([64, 128, 9, 9])
            flow = self.conv_blocks(out)  # torch.Size([64, 1, 36, 36])
            flow_flat = flow.view(flow.shape[0], -1)  # torch.Size([64, 1296])
            flow_out = self.out_layer(flow_flat)  # torch.Size([64, 36])

            return flow_out

            #1D





class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # input torch.Size([64, 1, 32, 32])
        if opt.dataset == 'mnist':
            '''
                    计算过程：[(32-3)+2]/2+1=15.5+1=15+1=16(下取整)
                    channels: 1->16
                    下面的同理
            '''
            self.model = nn.Sequential(
                #(64, 1, 32, 32)->(64, 16, 16, 16)
                *discriminator_block(opt.channels, 16, bn=False),
                *discriminator_block(16, 32),# (64, 16, 16, 16)->(64, 32, 8, 8)
                *discriminator_block(32, 64),# (64, 32, 8, 8)->(64, 64, 4, 4)
                *discriminator_block(64, 128),# (64, 64, 4, 4)->(64, 128, 2, 2)
            )

            # The height and width of downsampled image
            ds_size = opt.img_size // 2 ** 4
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        # input torch.Size([64, 1, 6, 6])
        elif opt.dataset == 'netflow':
            self.model = nn.Sequential(
                # (64, 36)->(64, 16, 16, 16)
                *discriminator_block(opt.channels, 3, bn=False),  # torch.Size([64, 3, 3, 3])
                *discriminator_block(3, 6),  # torch.Size([64, 6, 2, 2])
                *discriminator_block(6, 12),  # torch.Size([64, 12, 1, 1])
                # *discriminator_block(64, 128),  # (64, 64, 4, 4)->(64, 128, 2, 2)
            )

            # The height and width of downsampled image
            ds_size = opt.flow_size // 12 * 4
            self.adv_layer = nn.Sequential(nn.Linear(ds_size, 1), nn.Sigmoid())


    def forward(self, img):
        # input torch.Size([64, 1, 32, 32])
        if opt.dataset == 'mnist':
            out = self.model(img)# (64, 128, 2, 2)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)

        # input torch.Size([64, 36])
        elif opt.dataset == 'netflow':
            inputs = img.view(img.shape[0], 1, 6, 6)
            out = self.model(inputs)  # (64, 128, 2, 2)
            out = out.view(out.shape[0], -1)  # torch.Size([64, 12])
            validity = self.adv_layer(out)  # torch.Size([64, 1])

        return validity
