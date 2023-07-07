import os
import pickle
import torchvision.transforms as transforms

from gan4netflow.implementations.dataprocess4netflow import Dataset4netflow
from torchvision import datasets
from torch.autograd import Variable
from time import process_time

from models import *
from gan4netflow.implementations.config import *
# cuda = True if torch.cuda.is_available() else False
cuda = False
# 将训练过程中涉及的所有Tensor载入GPU中
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cpu")

# def get_random_data(real_samples, fake_samples, mode):
#     if mode == "mnist":
#         # 取一个随机权重，将真实/生成数据样本混合为加权样本，real_samples.size(0)就是batch_size
#         alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))  # torch.Size([1, 1, 1, 1])
#         # 根据随机权重生成混合样本
#         interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)  # torch.Size([64, 1, 28, 28])
#     elif mode == "netflow":
#         alpha = Tensor(np.random.random((real_samples.size(0), 1)))  # torch.Size([1, 1, 1, 1])
#         # 根据随机权重生成混合样本
#         interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#
#     return interpolates



if __name__ == '__main__':
    # test_data = {"ts":"2022-11-27T11:28:27.223249Z","uid":"CPHPed1o5fB4ucY6El","id.orig_h":"172.23.1.131","id.orig_p":62702,"id.resp_h":"224.0.0.252","id.resp_p":5355,"proto":"udp","service":"dns","duration":0.10699295997619629,"orig_bytes":74,"resp_bytes":0,"conn_state":"S0","local_orig":true,"local_resp":false,"missed_bytes":0,"history":"D","orig_pkts":2,"orig_ip_bytes":130,"resp_pkts":0,"resp_ip_bytes":0,"application2":"zeek"}
    # ----------
    #  获取数据
    # ----------


    testmodel_path = r"D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\模型保存\netflow\1005_model4netflow\221005discriminator_epoch_180.pth"
    print("载入模型参数")
    Discriminator_state_dict = torch.load(testmodel_path, map_location=torch.device('cpu'))
    # Generator_state_dict = torch.load(opt.model4netflow_G_path)
    discriminator = Discriminator()
    # generator = Generator()
    print("恢复模型参数")
    discriminator.load_state_dict(Discriminator_state_dict)
    # generator.load_state_dict(Generator_state_dict)

    if cuda:
        discriminator.cuda()
        # generator.cuda()

    print("Model Reload!")

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
            batch_size=1,# opt.batch_size
            shuffle=True,
            drop_last=True  # 若当前数据集中数据个数不是设置的batch_size的整数倍，丢弃当前batch防止报错
        )

    elif opt.dataset == 'netflow' and opt.testdata == 'null':
        dataprocess = Dataset4netflow()
        start = process_time()
        print("载入测试集数据")
        train_data = dataprocess.get_dataset(opt.testdata_path)
        end = process_time()
        loading_time = end - start
        print(f"=====Dataset is loaded, time:{loading_time}=====")
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                                 shuffle=True, drop_last=True)
    elif opt.dataset == 'netflow' and opt.testdata == 'es_flow':
        path4featuresincsv = r'D:\code\log4gan_dataset\zeeklog\22.9.19\conn_features9.19.csv'
        # 获取onehot编码器
        ohe_service = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_service.pkl', 'rb'))
        ohe_proto = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_proto.pkl', 'rb'))
        ohe_connstate = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_ohe_connstate.pkl', 'rb'))
        print("onehot编码器加载完成")

        # 载入scale参数与save_dict
        scaler = pickle.load(
            open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\important_dump\min_max_scaler.pkl',
                 'rb'))
        print("scale加载完成")
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
    else:
        raise NotImplementedError

    # 使用真实mnist/netflow数据进行测试
    if opt.testdata == 'es_flow':
        predict_flow = testrealsample_tensor.float()
        predict_samples = Variable(predict_flow)
        predict_validity, proba_predict = discriminator(predict_samples)
        # 计算概率均值
        probability_predict = torch.mean(proba_predict)
        print(f"该样本(from elasticsearch)正常的概率是：", probability_predict.data)
        print("\n")
    else:
        for i, (imgs, _) in enumerate(dataloader):
            """
            载入测试集数据作为real_samples，随机噪声作为对照数据（假数据/异常数据），imgs-->Tensor:(1,36)
            """
            if i <= 10:
                real_samples = Variable(imgs.type(Tensor))  # torch.Size([64, 1, 28, 28])/torch.Size([64, 36])/torch.Size([1, 36])
                real_validity, proba_real = discriminator(real_samples)

                # 计算概率均值
                probability_real = torch.mean(proba_real)
                print(f"该样本(real)来自{opt.dataset}数据集的概率是：", probability_real.data)

                if opt.dataset == 'mnist':
                    # # 随机tensor(与图片形状一致)
                    noise = Variable((torch.rand(1, 1, 28, 28)).cuda())
                elif opt.dataset == 'netflow':
                    # noise = Variable(Tensor(np.random.normal(0, 1, (real_samples.shape[0], opt.latent_dim4flow))))
                    # # noise.to(device)
                    # fake_flows = generator(noise)  # 构造假网络流
                    noise_flows = Variable((torch.randn(real_samples.size())).type(Tensor))

                # # 使用混合数据测试
                # # 获取真实数据与噪声数据的混合样本
                # mix_imgs = get_random_data(real_samples.data, noise_flows.data, mode=opt.dataset)
                # mix_validity, proba_mix = discriminator(mix_imgs)
                # # 计算概率均值
                # probability_mix = torch.mean(proba_mix)
                # print(f"该样本(mix)来自{opt.dataset}数据集的概率是：", probability_mix.data)

                #使用随机噪声测试
                noise_validity, proba_noise = discriminator(noise_flows)
                # 计算概率均值
                probability_noise = torch.mean(proba_noise)
                print(f"该样本(noise)来自{opt.dataset}数据集的概率是：", probability_noise.data)
                print("\n")
            else:
                break


