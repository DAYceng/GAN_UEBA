import sys, time, datetime
import pickle

from gan4netflow.implementations.datapreprocessing import Preprocessing
from gan4netflow.implementations.dataprocess4netflow import Dataset4netflow
from torch.autograd import Variable
import torch.autograd as autograd
from time import process_time
from elasticsearch import Elasticsearch

from gan4netflow.implementations.utils import Logger
from models import *
from gan4netflow.implementations.config import *

if opt.device == 'cpu':
    cuda = False
    device = torch.device("cpu")
elif opt.device == 'gpu':
    cuda = True
    device = torch.device('cuda')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples, mode):
    """计算 WGAN GP 的梯度惩罚损失"""
    # 取一个随机权重，将真实/生成数据样本混合为加权样本，real_samples.size(0)就是batch_size
    # setup_seed(1)
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))  # torch.Size([64, 1])
    # 根据随机权重生成混合样本
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)  # torch.Size([64, 14])
    d_interpolates, probability = D(interpolates)  # 判别混合样本
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # 判别器的判断结果d_interpolates与加权样本interpolates求导
    gradients = autograd.grad(
        outputs=d_interpolates,  # torch.Size([64, 1])
        inputs=interpolates,  # torch.Size([64, 14])
        grad_outputs=fake,  # 计算后输出形式与fake一致，为一个Tensor（Var）,torch.Size([64, 1])
        create_graph=True,  # 计算二阶导数
        retain_graph=True,  # 保留计算图
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # 对梯度做均方操作后取平均值得到gradient_penalty

    return gradient_penalty


def eval4model(real_output, generated_output):
    # 将真实样本的输出视为1，生成样本的输出视为0，然后根据阈值将输出转换为二进制分类结果
    threshold = 0.5
    real_binary_output = np.where(real_output > threshold, 1, 0)
    generated_binary_output = np.where(generated_output > threshold, 1, 0)

    # 计算判别器在真实样本和生成样本上的分类准确率
    real_accuracy = np.mean(real_binary_output)
    # generated_accuracy = np.mean(1 - generated_binary_output)

    return real_accuracy;



if __name__ == '__main__':
    if opt.operating_mode == 'detection':
        # ----------
        #  载入模型及参数
        # ----------
        testmodel_path = r"D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\模型保存\netflow\1005_model4netflow\221005discriminator_epoch_180.pth"
        print("载入模型参数")
        Discriminator_state_dict = torch.load(testmodel_path, map_location=torch.device('cpu'))
        discriminator = Discriminator()
        print("恢复模型参数")
        discriminator.load_state_dict(Discriminator_state_dict)

        if cuda:
            discriminator.cuda()

        print("Model Reload!")
        path4featuresincsv = r'D:\code\log4gan_dataset\zeeklog\22.9.19\conn_features9.19.csv'
        # 获取onehot编码器
        ohe_service = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_service.pkl', 'rb'))
        ohe_proto = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_proto.pkl', 'rb'))
        ohe_connstate = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_ohe_connstate.pkl', 'rb'))
        print("onehot编码器加载完成")

        # # 载入scale参数与save_dict
        # scaler = pickle.load(
        #     open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\important_dump\min_max_scaler.pkl',
        #          'rb'))
        # print("scale加载完成")
        # ----------
        #  获取数据
        # ----------
        # try:
        #     test_Preprocessing = Preprocessing()  # 载入数据预处理类
        #     es = Elasticsearch("http://172.23.6.213:9200")  # 创建es客户端
        #     data_from_es = Preprocessing.get_log_from_es(esclient=es)
        #     flow4predict = test_Preprocessing.get_predict_data(data_from_es, path4featuresincsv, ohe_service,
        #                                                        ohe_proto, ohe_connstate, scaler)
        #     predict_flow = torch.tensor(flow4predict, dtype=torch.float64)
        # except ConnectionError:
        #     print("es连接失败，接下来将使用本地单条数据进行测试")
        #     print("该数据来自测试集")
        #     predict_flow = torch.tensor([[9.2355e-09, 3.4912e-07, 1.0413e-07, 1.0000e+00, 0.0000e+00, 1.8894e-05,
        #                                        2.6675e-06, 2.0270e-05, 4.3418e-06, 1.0000e+00, 0.0000e+00, 1.0895e-02,
        #                                        0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
        #                                        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        #                                        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00]],
        #                                      dtype=torch.float64)
        testrealsample_tensor = torch.tensor([[9.2355e-09, 3.4912e-07, 1.0413e-07, 1.0000e+00, 0.0000e+00, 1.8894e-05,
                                               2.6675e-06, 2.0270e-05, 4.3418e-06, 1.0000e+00, 0.0000e+00, 1.0895e-02,
                                               0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
                                               0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                               0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                               0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00]],
                                             dtype=torch.float64)
        # ----------
        #  检测
        # ----------
        predict_samples = Variable(testrealsample_tensor.type(Tensor)) if cuda else Variable(
            testrealsample_tensor.float())
        predict_validity, proba_predict = discriminator(predict_samples)
        # 计算概率均值
        probability_predict = torch.mean(proba_predict)
        print(f"该样本(from elasticsearch)正常的概率是：", probability_predict.data)
        print("\n")

    elif opt.operating_mode == 'train':
        sys.stdout = Logger("WGAN-GP", opt.dataset, stream=sys.stdout)
        t = datetime.datetime.now()
        savetime = t.strftime('%y%m%d')
        print('后面所有在控制台打印的内容，将会保存到Log文件中')

        # 初始化generator和discriminator
        generator = Generator()
        discriminator = Discriminator()

        path4featuresincsv = r'D:\code\log4gan_dataset\zeeklog\22.9.19\conn_features9.19.csv'
        # 获取onehot编码器
        ohe_service = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_service.pkl', 'rb'))
        ohe_proto = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_proto.pkl', 'rb'))
        ohe_connstate = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_ohe_connstate.pkl', 'rb'))
        print("onehot编码器加载完成")

        if cuda:
            generator.cuda()
            discriminator.cuda()

        # ---------------------
        #  载入训练数据
        # ---------------------
        dataprocess = Dataset4netflow()
        start = process_time()

        test_Preprocessing = Preprocessing()  # 载入数据预处理类
        es = Elasticsearch("http://172.23.6.213:9200")  # 创建es客户端
        data_from_es = Preprocessing.get_log_from_es(esclient=es)
        esdata = test_Preprocessing.get_predict_data(data_from_es, path4featuresincsv, ohe_service,
                                                           ohe_proto, ohe_connstate)
        train_data = dataprocess.get_dataset(esdata)
        end = process_time()
        loading_time = end - start
        print(f"=====Dataset is loaded, time:{loading_time}=====")
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                                 shuffle=False, drop_last=True)

        # 指定优化器
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.lr)


        batches_done = 0
        # 设定梯度惩罚的权重
        lambda_gp = 10
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
                z = Variable(
                    Tensor(np.random.normal(0, 1, (real_flows.shape[0], opt.latent_dim4flow))))  # torch.Size([650])
                # 调用生成器输出假网络流，torch.Size([64, 14])
                fake_flows = generator(z)
                # 判别器判断真实网络流
                real_validity, probability = discriminator(real_flows)
                # 判别器判断假网络流
                fake_validity, probability = discriminator(fake_flows)
                # 计算梯度惩罚值
                gradient_penalty = compute_gradient_penalty(discriminator, real_flows.data, fake_flows.data,
                                                            mode='netflow')
                # 评估当前模型判别器的acc
                if epoch % 20 == 0:
                    real_accuracy = eval4model(real_validity, fake_validity);
                    print(real_accuracy)
                    print(torch.mean(probability))

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

            # if epoch % 20 == 0:
            #     torch.save(discriminator.state_dict(),
            #                f'%s\\{savetime}discriminator_epoch_%d.pth' % (opt.save_model4netflow_dir, epoch))
            #     torch.save(generator.state_dict(),
            #                f'%s\\{savetime}generator_epoch_%d.pth' % (opt.save_model4netflow_dir, epoch))

            batch_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(batch_time)
