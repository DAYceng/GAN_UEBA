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
# test_dict = {"ts":"2022-11-27T11:28:25.153860Z","uid":"Cry0PW1EqRFZuReYMb","id.orig_h":"172.23.6.216","id.orig_p":38938,"id.resp_h":"172.23.6.214","id.resp_p":1514,"proto":"tcp","service":"unknown","duration":4.9954218864440918,"orig_bytes":1180,"resp_bytes":178,"conn_state":"RSTR","local_orig":1,"local_resp":1,"missed_bytes":0,"history":"ShADadfr","orig_pkts":8,"orig_ip_bytes":1604,"resp_pkts":8,"resp_ip_bytes":590,"application2":"zeek"}
hits_list = Preprocessing.get_log_from_es(esclient=es)

# print(hits_list)


# test_jsonlist = json.dumps(hits_list)
deal_jsonlist = test_Preprocessing.extract_five_tup(hits_list, ohe_service, ohe_proto, ohe_connstate)
print(deal_jsonlist)