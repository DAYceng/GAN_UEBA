import torch

from gan4netflow.implementations.datapreprocessing import Preprocessing
from elasticsearch import Elasticsearch
import pickle
from torch.autograd import Variable

test_Preprocessing = Preprocessing()
es = Elasticsearch("http://172.23.6.213:9200")
hits_list = Preprocessing.get_log_from_es(esclient=es)
path4featuresincsv = r'D:\code\log4gan_dataset\zeeklog\22.9.19\conn_features9.19.csv'



# 获取onehot编码器
ohe_service = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_service.pkl', 'rb'))
ohe_proto = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_proto.pkl', 'rb'))
ohe_connstate = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_ohe_connstate.pkl', 'rb'))

# 载入scale参数与save_dict
scaler = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\important_dump\min_max_scaler.pkl', 'rb'))
# scaler = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\important_dump\all_data_minmaxscaler.pkl', 'rb'))
# saved_dict = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\important_dump\saved_dict.pkl', 'rb'))
flow_in_dictlist = test_Preprocessing.extract_five_tup(hits_list, ohe_service, ohe_proto, ohe_connstate)
flow2df = test_Preprocessing.data_processing(flow_in_dictlist, path4featuresincsv)

data_minmaxined = test_Preprocessing.feature_engineering(flow2df, ohe_service, ohe_proto, ohe_connstate)
print(data_minmaxined)

