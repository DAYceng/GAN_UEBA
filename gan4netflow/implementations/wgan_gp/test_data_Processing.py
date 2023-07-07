from gan4netflow.implementations.datapreprocessing import Preprocessing
from elasticsearch import Elasticsearch
import json
import pickle

test_Preprocessing = Preprocessing()
es = Elasticsearch("http://172.23.6.213:9200")
# 获取onehot编码器
ohe_service = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_service.pkl', 'rb'))
ohe_proto = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_proto.pkl', 'rb'))
ohe_connstate = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_ohe_connstate.pkl', 'rb'))
path4featuresincsv = r'D:\code\log4gan_dataset\zeeklog\22.9.19\conn_features9.19.csv'
hits_list = Preprocessing.get_log_from_es(esclient=es)
# test_json = json.dumps(test_dict)
flow_in_dict = test_Preprocessing.extract_five_tup(hits_list, ohe_service, ohe_proto, ohe_connstate)
flow2df = test_Preprocessing.data_processing(flow_in_dict, path4featuresincsv)
print(flow2df)