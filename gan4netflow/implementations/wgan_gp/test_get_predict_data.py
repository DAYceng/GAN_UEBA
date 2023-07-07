from elasticsearch import Elasticsearch
import pickle
from gan4netflow.implementations.datapreprocessing import Preprocessing

path4featuresincsv = r'D:\code\log4gan_dataset\zeeklog\22.9.19\conn_features9.19.csv'
# 获取onehot编码器
ohe_service = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_service.pkl', 'rb'))
ohe_proto = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_proto.pkl', 'rb'))
ohe_connstate = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_ohe_connstate.pkl', 'rb'))

# 载入scale参数与save_dict
scaler = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\important_dump\min_max_scaler.pkl', 'rb'))

# parameter file
test_Preprocessing = Preprocessing()  # 载入数据预处理类
es = Elasticsearch("http://172.23.6.213:9200")  # 创建es客户端
data_from_es = Preprocessing.get_log_from_es(esclient=es)
# 有bug，遇到一些异常查询值会报错
# 在extract_five_tup增加判定规则，如果出现在训练过程中没有的字段内容，删除对应行
# print(flow2df["xxx"].value_counts().index)  # 查看英语列有多少不同的分数
flow4predict = test_Preprocessing.get_predict_data(data_from_es, path4featuresincsv, ohe_service,
                                                   ohe_proto, ohe_connstate, scaler)
print(flow4predict)