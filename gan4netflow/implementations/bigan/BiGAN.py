import os, sys, time, datetime

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from time import process_time

from gan4netflow.implementations.bigan.models import *
from gan4netflow.implementations.dataprocess4netflow import Dataset4netflow
from gan4netflow.implementations.utils import Logger, draw_splicingimgs

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("images", exist_ok=True)

# os.makedirs("images_cifar10", exist_ok=True)
sys.stdout = Logger("BiGAN", opt.dataset, stream=sys.stdout)
t = datetime.datetime.now()
savetime = t.strftime('%y%m%d')
print('后面所有在控制台打印的内容，将会保存到Log文件中')

if opt.dataset == 'mnist':
    # 配置data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    start = process_time()
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        # drop_last=True #若当前数据集中数据个数不是设置的batch_size的整数倍，丢弃当前batch防止报错
    )
    end = process_time()
    loading_time = end - start
    print(f"=====Dataset is loaded, time:{loading_time}=====")
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

def D_loss(DG, DE, eps=1e-6):
    loss = torch.log(DE + eps) + torch.log(1 - DG + eps)
    return -torch.mean(loss)

def EG_loss(DG, DE, eps=1e-6):
    loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    return -torch.mean(loss)

def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)

n_epochs = 400
l_rate = 2e-5

encoder = Encoder().to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

encoder.apply(init_weights)
generator.apply(init_weights)
discriminator.apply(init_weights)

#optimizers with weight decay
optimizer_EG = torch.optim.Adam(list(encoder.parameters()) + list(generator.parameters()),
                                lr=l_rate, betas=(0.5, 0.999), weight_decay=1e-5)
optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=l_rate, betas=(0.5, 0.999), weight_decay=1e-5)

#optimizers without weight decay
# optimizer_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=l_rate, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(D.parameters(), lr=l_rate, betas=(0.5, 0.999))

#learning rate scheduler
# lambda_ = lambda epoch: 1 if epoch < 300 else 0.978 ** (epoch-300)
# scheduler_EG = torch.optim.lr_scheduler.LambdaLR(optimizer_EG, lr_lambda=lambda_)
# scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_)


# 将训练过程中涉及的所有Tensor载入GPU中
Tensor = torch.cuda.FloatTensor  # torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    D_loss_acc = 0.
    EG_loss_acc = 0.
    discriminator.train()
    encoder.train()
    generator.train()

    #     scheduler_D.step()
    #     scheduler_EG.step()

    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        imgs = imgs.reshape(imgs.size(0), -1)

        # 生成标签的onehot编码（10维）
        c = torch.zeros(imgs.size(0), 10, dtype=torch.float32).to(device)
        c[torch.arange(imgs.size(0)), labels] = 1

        # initialize z from 50-dim U[-1,1], 生成50维的随机噪声
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))) - 1

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # 生成器生成假样本，编码器提取真实数据特征
        Gz = generator(z, c)  # torch.Size([128, 784])
        EX = encoder(imgs)  # torch.Size([128, 50])

        # 输入生成器样本
        DG = discriminator(Gz, z, c)  # torch.Size([128, 1])
        # 输入编码器样本
        DE = discriminator(imgs, EX, c)  # torch.Size([128, 1])

        # 计算判别器loss
        loss_D = D_loss(DG, DE)
        D_loss_acc += loss_D.item()

        loss_D.backward(retain_graph=True)
        optimizer_D.step()
        # --------------------------
        #  Train Encoder & Generator
        # --------------------------
        optimizer_EG.zero_grad()

        # 重复写两遍的原因是防止在计算Encoder和Generator的loss时出错（因为计算前要现将当前梯度清空）
        Gz = generator(z, c)
        EX = encoder(imgs)


        DG = discriminator(Gz, z, c)
        DE = discriminator(imgs, EX, c)


        loss_EG = EG_loss(DG, DE)
        EG_loss_acc += loss_EG.item()

        loss_EG.backward()
        optimizer_EG.step()

        if i > 0 and i % 5 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [Avg_Loss_D: %f] [Avg_Loss_EG: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), D_loss_acc / i, EG_loss_acc / i)
            )

    if (epoch + 1) % 10 == 0:
        n_show = 10
        discriminator.eval()
        encoder.eval()
        generator.eval()

        with torch.no_grad():
            # 切断梯度累计并生成假图片
            real = imgs[:n_show]
            c = torch.zeros(n_show, 10, dtype=torch.float32).to(device)
            c[torch.arange(n_show), labels[:n_show]] = 1
            z = Variable(Tensor(np.random.normal(0, 1, (n_show, opt.latent_dim)))) - 1

            gener = generator(z, c)  # tensor torch.Size([28, 28])
            recon = generator(encoder(real), c)  # torch.Size([28, 28])
            real = real.reshape(n_show, 1, 28, 28)

            if opt.dataset == 'mnist':
                draw_splicingimgs(gener, real, recon, n_show, epoch, save_samples=True)

    if epoch % 20 == 0:
        torch.save(discriminator.state_dict(), '%s\\discriminator_epoch_%d.pth' % (opt.save_model4mnist_dir, epoch))
        torch.save(generator.state_dict(), '%s\\generator_epoch_%d.pth' % (opt.save_model4mnist_dir, epoch))
        torch.save(encoder.state_dict(), '%s\\encoder_epoch_%d.pth' % (opt.save_model4mnist_dir, epoch))
#save model
torch.save({
            'D_state_dict': discriminator.state_dict(),
            'E_state_dict': encoder.state_dict(),
            'G_state_dict': generator.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'optimizer_EG_state_dict': optimizer_EG.state_dict(),
            #'scheduler_D_state_dict': scheduler_D.state_dict(),
            #'scheduler_EG_state_dict': scheduler_EG.state_dict()
            }, r'D:\code\gan4netflow\implementations\bigan\modelsave\model4training\model4mnist\models_state_dict_CBiGAN.tar')

# save final results
n_show = 20
discriminator.eval()
encoder.eval()
generator.eval()

fig, ax = plt.subplots(3, n_show, figsize=(25, 5))
fig.subplots_adjust(wspace=0.0, hspace=0)
plt.rcParams.update({'font.size': 30})
fig.text(0.04, 0.75, 'G(z, c)', ha='left')
fig.text(0.04, 0.5, 'Real', ha='left')
fig.text(0.04, 0.25, 'G(E(X), c)', ha='left')

with torch.no_grad():
    for i in range(10):
        real = imgs[labels == i][:2]

        c = torch.zeros(2, 10, dtype=torch.float32).to(device)
        c[torch.arange(2), i] = 1
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        gener = generator(z, c).reshape(2, 28, 28).cpu().numpy()
        recon = generator(encoder(real), c).reshape(2, 28, 28).cpu().numpy()
        real = real.reshape(2, 28, 28).cpu().numpy()

        ax[0, i].imshow(gener[0], cmap='gray')
        ax[0, i].axis('off')
        ax[0, i + 10].imshow(gener[1], cmap='gray')
        ax[0, i + 10].axis('off')

        ax[1, i].imshow(real[0], cmap='gray')
        ax[1, i].axis('off')
        ax[1, i + 10].imshow(real[1], cmap='gray')
        ax[1, i + 10].axis('off')

        ax[2, i].imshow(recon[0], cmap='gray')
        ax[2, i].axis('off')
        ax[2, i + 10].imshow(recon[1], cmap='gray')
        ax[2, i + 10].axis('off')

    plt.savefig(r'D:\code\gan4netflow\implementations\bigan\images\Final_Results_CBiGAN.png')