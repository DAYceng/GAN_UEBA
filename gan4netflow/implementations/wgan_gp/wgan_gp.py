import os, sys, time, datetime
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from time import process_time
from gan4netflow.implementations.dataprocess4netflow import Dataset4netflow
from models import *
from config import *
from gan4netflow.implementations.utils import Logger, setup_seed


os.makedirs("images", exist_ok=True)

# os.makedirs("images_cifar10", exist_ok=True)
sys.stdout = Logger("WGAN-GP", opt.dataset, stream=sys.stdout)
t = datetime.datetime.now()
savetime = t.strftime('%y%m%d')
print('后面所有在控制台打印的内容，将会保存到Log文件中')


cuda = True if torch.cuda.is_available() else False


# 设定梯度惩罚的权重
lambda_gp = 10

# 初始化generator和discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

device = torch.device('cuda')

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

# 指定优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.lr)
# optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

# 将训练过程中涉及的所有Tensor载入GPU中
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples, mode):
    """计算 WGAN GP 的梯度惩罚损失"""
    if mode == 'mnist':
        # 取一个随机权重，将真实/生成数据样本混合为加权样本，real_samples.size(0)就是batch_size
        setup_seed(1)
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))  # torch.Size([64, 1, 1, 1])
        # 根据随机权重生成混合样本
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
            True)  # torch.Size([64, 1, 28, 28])
        d_interpolates, probability = D(interpolates)  # 判别混合样本
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

    if mode == 'netflow':
        # 取一个随机权重，将真实/生成数据样本混合为加权样本，real_samples.size(0)就是batch_size
        # setup_seed(1)
        alpha = Tensor(np.random.random((real_samples.size(0), 1)))# torch.Size([64, 1])
        # 根据随机权重生成混合样本
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)# torch.Size([64, 14])
        d_interpolates, probability = D(interpolates)# 判别混合样本
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # 判别器的判断结果d_interpolates与加权样本interpolates求导
    gradients = autograd.grad(
        outputs=d_interpolates,# torch.Size([64, 1])
        inputs=interpolates,# torch.Size([64, 14])
        grad_outputs=fake,# 计算后输出形式与fake一致，为一个Tensor（Var）,torch.Size([64, 1])
        create_graph=True,# 计算二阶导数
        retain_graph=True,# 保留计算图
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()# 对梯度做均方操作后取平均值得到gradient_penalty
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
if opt.dataset == 'mnist':
    for epoch in range(opt.n_epochs):
        for step, (imgs, _) in enumerate(dataloader):
            # 将真实图片转换为Variable备用，torch.Size([64, 3, 32, 32])
            real_imgs = Variable(imgs.type(Tensor))# torch.Size([64, 1, 28, 28])
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # 生成100维的随机噪声
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))# torch.Size([64, 100])
            # 调用生成器输出假图片，torch.Size([64, 3, 32, 32])
            fake_imgs = generator(z)# torch.Size([64, 1, 28, 28])
            # 判别器判断真实图片
            real_validity, probability = discriminator(real_imgs)
            # 判别器判断假图片
            fake_validity, probability = discriminator(fake_imgs)
            # 计算梯度惩罚值
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, mode=opt.dataset)
            # 求损失值
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            # 每训练五次判别器训练一次生成器
            if step % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # 生成假图片
                fake_imgs = generator(z)
                # 此处的损失用于衡量生成器欺骗鉴别器的能力
                fake_validity, probability = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, step, len(dataloader), d_loss.item(), g_loss.item())
                )

                if batches_done % opt.sample_interval == 0:
                    if opt.dataset == 'mnist':
                        save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += opt.n_critic

        if epoch % 20 == 0:
            torch.save(discriminator.state_dict(), '%s\\discriminator_epoch_%d.pth' % (opt.save_model4mnist_dir, epoch))
            torch.save(generator.state_dict(), '%s\\generator_epoch_%d.pth' % (opt.save_model4mnist_dir, epoch))

elif opt.dataset == 'netflow':
    for epoch in range(opt.n_epochs):
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            flows, _ = batch
            real_flows = Variable(flows.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # 生成18维的随机噪声
            # setup_seed(1)
            z = Variable(Tensor(np.random.normal(0, 1, (real_flows.shape[0], opt.latent_dim4flow))))  # torch.Size([650])
            # 调用生成器输出假网络流，torch.Size([64, 14])
            fake_flows = generator(z)
            # 判别器判断真实网络流
            real_validity, probability = discriminator(real_flows)
            # 判别器判断假网络流
            fake_validity, probability = discriminator(fake_flows)
            # 计算梯度惩罚值
            gradient_penalty = compute_gradient_penalty(discriminator, real_flows.data, fake_flows.data, mode='netflow')
            # 求损失值
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            # 每训练五次判别器训练一次生成器opt.n_critic
            if step % 5 == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # 生成假网络流
                fake_flows = generator(z)
                # 此处的损失用于衡量生成器欺骗鉴别器的能力
                fake_validity, probability = discriminator(fake_flows)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()


                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, step, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done += opt.n_critic

        if epoch % 20 == 0:
            torch.save(discriminator.state_dict(), f'%s\\{savetime}discriminator_epoch_%d.pth' % (opt.save_model4netflow_dir, epoch))
            torch.save(generator.state_dict(), f'%s\\{savetime}generator_epoch_%d.pth' % (opt.save_model4netflow_dir, epoch))

        batch_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(batch_time)

