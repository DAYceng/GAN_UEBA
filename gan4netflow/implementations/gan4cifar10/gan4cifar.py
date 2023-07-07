import torch.nn as nn
import torch.nn.init as init
import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False


latent_size = 64
n_channel = 3
n_g_feature = 64

# 输入形状：torch.Size([64, 3, 32, 32])
# 计算方法：
    # 输出大小 = (输入大小-1)x步长+2*padding+(卷积核-1)+1
gnet = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
    nn.BatchNorm2d(4 * n_g_feature),
    nn.ReLU(),# (32-1)x1+2*0+(4-1)+1=35

    nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_g_feature),
    nn.ReLU(),# (35-1)x2+2*1+(4-1)+1=74

    nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(n_g_feature),
    nn.ReLU(),# (74-1)x2+2*1+(4-1)+1=152

    nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4, stride=2, padding=1),
    nn.Sigmoid()# (152-1)x2+2*1+(4-1)+1=308
)


n_d_feature = 64
dnet = nn.Sequential(
    nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_d_feature),
    nn.LeakyReLU(0.2),

    nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * n_d_feature),
    nn.LeakyReLU(0.2),

    nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
)


def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)

gnet.apply(weights_init)
dnet.apply(weights_init)

if cuda:
    gnet.cuda()
    dnet.cuda()

# 将训练过程中涉及的所有Tensor载入GPU中
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def tocuda(x):
    if cuda:
        return x.cuda()
    return x


dataset = CIFAR10(root='./CIFARdata', download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


criterion = nn.BCEWithLogitsLoss()
goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))


batch_size = 64
fixed_noises = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))

epoch_num = 15000
for epoch in range(epoch_num):
    for batch_idx, (data, _) in enumerate(dataloader):
        # 将真实图片转换为Variable备用，torch.Size([64, 3, 32, 32])
        real_images = Variable(data.type(Tensor))
        batch_size = real_images.size(0)

        real_labels = Variable(tocuda(torch.ones(batch_size)))#真实标签
        real_validity = dnet(real_images)
        outputs = real_validity.reshape(-1)
        dloss_real = criterion(outputs, real_labels)
        dmean_real = outputs.sigmoid().mean()

        noises = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        fake_images = gnet(noises)
        fake_labels = Variable(tocuda(torch.zeros(batch_size)))
        fake = fake_images.detach()

        fake_validity = dnet(fake)
        outputs = fake_validity.view(-1)
        dloss_fake = criterion(outputs, fake_labels)
        dmean_fake = outputs.sigmoid().mean()

        dloss = dloss_real + dloss_fake
        dnet.zero_grad()
        dloss.backward()
        doptimizer.step()


        labels = Variable(tocuda(torch.ones(batch_size)))
        preds = dnet(fake_images)
        outputs = preds.view(-1)
        gloss = criterion(outputs, labels)
        gmean_fake = outputs.sigmoid().mean()
        gnet.zero_grad()
        gloss.backward()
        goptimizer.step()

        if batch_idx % 100 == 0:
            fake = gnet(fixed_noises)
            save_image(fake, f'./GAN_saved02/images_epoch{epoch:02d}_batch{batch_idx:03d}.png')

            print(f'Epoch index: {epoch}, {epoch_num} epoches in total.')
            print(f'Batch index: {batch_idx}, the batch size is {batch_size}.')
            print(f'Discriminator loss is: {dloss}, generator loss is: {gloss}', '\n',
                  f'Discriminator tells real images real ability: {dmean_real}', '\n',
                  f'Discriminator tells fake images real ability: {dmean_fake:g}/{gmean_fake:g}')


gnet_save_path = 'gnet.pt'
torch.save(gnet, gnet_save_path)
# gnet = torch.load(gnet_save_path)
# gnet.eval()

dnet_save_path = 'dnet.pt'
torch.save(dnet, dnet_save_path)
# dnet = torch.load(dnet_save_path)
# dnet.eval()

for i in range(100):
    noises = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
    fake_images = gnet(noises)
    save_image(fake, f'./test_GAN/{i}.png')

# print(gnet, dnet)

