testlist = ['OTH' 'REJ' 'RSTO' 'RSTR' 'RSTRH' 'S0' 'S1' 'S2' 'S3' 'SF' 'SHR']
nornmallist = ['dns', 'udp', 'OTH']
assholelist = ['dns', 'udp', 'CC']
# ziduan = 'CC'
# if ziduan not in testlist:
#     print(1)
from gan4netflow.implementations.datapreprocessing import Preprocessing
from elasticsearch import Elasticsearch
import pickle
test_Preprocessing = Preprocessing()

# 获取onehot编码器
ohe_service = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_service.pkl', 'rb'))
ohe_proto = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_proto.pkl', 'rb'))
ohe_connstate = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_ohe_connstate.pkl', 'rb'))


print(test_Preprocessing._check_odd(nornmallist, ohe_service, ohe_proto, ohe_connstate))