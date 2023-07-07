import os, sys, time, datetime

from time import process_time

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from implementations.dataprocess4netflow import Dataset4netflow
from implementations.dcgan.models import *
from implementations.utils import Logger

os.makedirs("images", exist_ok=True)
sys.stdout = Logger("WGAN-GP", opt.dataset, stream=sys.stdout)
t = datetime.datetime.now()
savetime = t.strftime('%y%m%d')
print('后面所有在控制台打印的内容，将会保存到Log文件中')

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

if opt.dataset == 'mnist':
    # 配置data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True #若当前数据集中数据个数不是设置的batch_size的整数倍，丢弃当前batch防止报错
    )
elif opt.dataset == 'netflow':
    dataprocess = Dataset4netflow()
    start = process_time()
    train_data = dataprocess.get_dataset(opt.traindata_path)
    end = process_time()
    loading_time = end - start
    print(f"=====Dataset is loaded, time:{loading_time}=====")
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                   shuffle=False, drop_last=True)
else:
    raise NotImplementedError

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
if opt.dataset == 'mnist':
    for epoch in range(opt.n_epochs):
        for i, (samples, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(samples.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (samples.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i

            if epoch % 20 == 0:
                torch.save(discriminator.state_dict(), f'%s\\{savetime}discriminator_epoch_%d.pth' % (opt.save_model4netflow_dir, epoch))
                torch.save(generator.state_dict(), f'%s\\{savetime}generator_epoch_%d.pth' % (opt.save_model4netflow_dir, epoch))

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

elif opt.dataset == 'netflow':
    for epoch in range(opt.n_epochs):
        for i, (samples, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(samples.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (samples.shape[0], opt.latent_dim4flow))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i

            if epoch % 20 == 0:
                torch.save(discriminator.state_dict(),
                           f'%s\\{savetime}discriminator_epoch_%d.pth' % (opt.save_model4netflow_dir, epoch))
                torch.save(generator.state_dict(),
                           f'%s\\{savetime}generator_epoch_%d.pth' % (opt.save_model4netflow_dir, epoch))

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)



batch_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print(batch_time)