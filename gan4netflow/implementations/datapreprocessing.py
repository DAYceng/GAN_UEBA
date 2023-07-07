import json
import pandas as pd  # for csv files and dataframe
import warnings
import numpy

import torch
from torch.utils.data import TensorDataset

warnings.filterwarnings("ignore")

from gan4netflow.implementations.config import *

def scaling(low4scaling, upper4scaling, dataframe_row, scaling_mode):
    '''
    将给定的df列数据缩放都指定范围
    :param uplimit: 缩放范围上限
    :param donwlimit: 缩放范围下限
    :param dataframe_row: 目标dataframe中待处理的列
    :param scaling_mode: 缩放模式，'train' | 'detection'
    :return:
    '''
    if scaling_mode == 'train':
        # 找出列中的最值
        row_max = dataframe_row.max()
        row_min = dataframe_row.min()
        if row_min == row_max == 0:
            return dataframe_row
        dataframe_row = (dataframe_row - row_min) / (row_max - row_min) * (
                upper4scaling - low4scaling) + low4scaling
    elif scaling_mode == 'detection':
        # min_value = 0
        # max_value = 100
        # dataframe_row = (dataframe_row - min_value) / (max_value - min_value)
        if dataframe_row.max() > 0 and dataframe_row.max() < 1:
            return dataframe_row
        elif dataframe_row.max() > 1 and dataframe_row.max() < 10:
            min_value = 0
            max_value = 10
            dataframe_row = (dataframe_row - 0) / (max_value - 0)
        elif dataframe_row.max() > 10 and dataframe_row.max() < 100:
            min_value = 0
            max_value = 100
            dataframe_row = (dataframe_row - 0) / (max_value - 0)
        elif dataframe_row.max() > 100 and dataframe_row.max() < 1000:
            min_value = 0
            max_value = 1000
            dataframe_row = (dataframe_row - 0) / (max_value - 0)
        elif dataframe_row.max() > 1000:
            min_value = 0
            max_value = 1000
            dataframe_row = (dataframe_row - 0) / (max_value - 0)
            while(dataframe_row.max() > 1):
                dataframe_row = dataframe_row/10

    return dataframe_row

def patch_field(flow_dict):
    field_template = ['id.orig_h', 'id.resp_h', 'id.resp_p',
                      'proto', 'service', 'duration', 'orig_bytes',
                      'resp_bytes', 'conn_state', 'local_orig',
                      'local_resp', 'orig_pkts', 'orig_ip_bytes',
                      'resp_pkts', 'resp_ip_bytes', 'port_dynamic',
                      'port_static', 'access_RSID']
    if(len(flow_dict) == len(field_template)):
        return flow_dict
    else:
        # 获取当前flow的所有字段
        flowfield_list = list(flow_dict.keys())
        # 遍历字段模板，重构flow
        fix_flow_dict = {}
        for i, field in enumerate(field_template):
            # 某个字段在当前flow中不存在，则以0为值补充
            if field not in flowfield_list and field == 'service':
                fix_flow_dict[field] = 'unknown'
            elif field not in flowfield_list:
                fix_flow_dict[field] = 0
            else:
                fix_flow_dict[field] = flow_dict[field]
        return fix_flow_dict



class Preprocessing:
    def __init__(self):
        pass
        # self.mode = 'detection'

    @staticmethod
    def get_log_from_es(esclient, search_size, period="now-1y"):
        """
            对进行初步数据清洗后的数据进行标准化处理（含离散数据编码）
            :param esclient:elasticsearch clint by Python API
            :param period:es查询时间范围
        ##### 年:now-1y,月:now-1m,周:now-1w,日:now-1d,小时:now-1h,分钟:now-1m,秒now-1s #####
            :return: 在当前范围下查询到的最新一条数据
        """
        hits_list = []
        res = esclient.search(index='zeek-*', body={
            "size": search_size,
            "aggs": {
                "frequent_tags": {
                    "terms": {
                        "size": 95000,
                        "field": "full_log.keyword",
                    }
                }
            },
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": period,
                                    "lt": "now"
                                }
                            }
                        },
                        {
                            "bool": {
                                "must": {
                                    "exists": {
                                        "field": "data.conn_state.keyword"
                                    }}
                            }
                        }]
                }
            }
        })

        for hits in res['hits']['hits']:
            temp = json.loads(hits['_source']['full_log'])
            hits_deal = json.dumps(temp)

            hits_list.append(hits_deal)
            # print(hits['_source']['full_log'])

        return hits_list

    def get_predict_data(self, flow_datalist, featuresincsv,
                         ohe_service, ohe_proto, ohe_connstate):

        flow_in_dict = self.extract_five_tup(flow_datalist, ohe_service, ohe_proto, ohe_connstate)
        flow2df = self.data_processing(flow_in_dict, featuresincsv)
        data_minmaxined = self.feature_engineering(flow2df, ohe_service, ohe_proto,
                                                     ohe_connstate)

        return data_minmaxined

    def extract_five_tup(self, hits_list, ohe_service, ohe_proto, ohe_connstate):
        """
        对json格式的数据进行处理，计算访问关系，删除多余字段
            :param hits_list:从es查询到的json数据列表
            :return: deal_jsonlist:处理后的json列表
        """

        deal_jsonlist = []
        for flow_in_json in hits_list:
            line2dict = json.loads(flow_in_json)
            # 获取五元组
            src_ip = json.loads(str(flow_in_json))['id.orig_h']
            dst_ip = json.loads(str(flow_in_json))['id.resp_h']
            src_port = json.loads(str(flow_in_json))['id.orig_p']  # int
            dst_port = json.loads(str(flow_in_json))['id.resp_p']  # int

            # 将源IP、目的IP、目的端口以及协议类型转换为整数,即访问关系ID
            intsrc_ip = self._addr2int(src_ip)
            # 将目的端口按动(0-1023)/静(1024-65535)态进行分类编码
            port_dynamic, port_static = self._porttype_encode(dst_port)
            line2dict['port_dynamic'] = port_dynamic
            line2dict['port_static'] = port_static

            # # 忽略ipv6地址
            # if intsrc_ip == False:
            #     continue
            intdst_ip = self._addr2int(dst_ip)
            # 在计算access_RSID时，有可能会遇到那种比较长的端口，导致计算结果偏大
            # 如果遇到ipv6地址直接使access_RSID为65535,因为其地址长度为ipv4的4倍，按最大值(FFFF,即65535)算
            if intsrc_ip == intdst_ip == 65535:
                access_RSID = 65535
                # print(f"The address:{src_port} is an IPv6 address")
            else:
                access_RSID = intsrc_ip + intdst_ip + dst_port
            # 因为会有部分网络流的访问关系再计算后大小超过65535
            # 故将access_rsid缩放至0~65535的范围内
            # 这样不会影响ipv6地址的访问关系值，因为其缩放后仍为65535
            low4scaling, upper4scaling = 0, 65535
            access_rsid65535 = access_RSID % (upper4scaling - low4scaling + 1) + low4scaling

            line2dict["access_RSID"] = access_rsid65535
            del line2dict['ts']
            del line2dict['uid']
            del line2dict['id.orig_p']
            if 'history' in line2dict.keys():
                del line2dict['history']
            del line2dict['missed_bytes']
            line2dict.pop('application2')  # 删除zeek多余字段'application2'

            line2dict_patched = patch_field(line2dict)

            check_datalist = []
            proto = line2dict_patched['proto']
            service = line2dict_patched['service']
            conn_state = line2dict_patched['conn_state']
            check_datalist.append(service)
            check_datalist.append(proto)
            check_datalist.append(conn_state)

            # 检查数据中是否存在训练时不存在的异常字段值
            if self._check_odd(check_datalist, ohe_service, ohe_proto, ohe_connstate):
                continue
            else:
                deal_jsonlist.append(line2dict_patched)

            # if len(line2dict) == 18:  # 检查数据中字段数量是否与训练时一致
            #     check_datalist = []
            #     proto = json.loads(str(flow_in_json))['proto']
            #     service = json.loads(str(flow_in_json))['service']
            #     conn_state = json.loads(str(flow_in_json))['conn_state']
            #     check_datalist.append(service)
            #     check_datalist.append(proto)
            #     check_datalist.append(conn_state)
            #
            #     # 检查数据中是否存在训练时不存在的异常字段值
            #     if self._check_odd(check_datalist, ohe_service, ohe_proto, ohe_connstate):
            #         continue
            #     deal_jsonlist.append(line2dict)
            # # print(line2dict)

        return deal_jsonlist

    def _check_odd(self, check_list, ohe_service, ohe_proto, ohe_connstate):
        '''
        测试例子
        test_Preprocessing = Preprocessing()

        # 获取onehot编码器
        ohe_service = pickle.load(open(r'D:\...\ohe_service.pkl', 'rb'))
        ohe_proto = pickle.load(open(r'D:\...\ohe_proto.pkl', 'rb'))
        ohe_connstate = pickle.load(open(r'D:\...\ohe_ohe_connstate.pkl', 'rb'))

        nornmallist = ['dns', 'udp', 'OTH']
        assholelist = ['dns', 'udp', 'CC']
        print(test_Preprocessing._check_odd(nornmallist, ohe_service, ohe_proto, ohe_connstate))
        '''
        count = 0
        for checkdata, ohe in zip(check_list, [ohe_service, ohe_proto, ohe_connstate]):
            if checkdata not in ohe.categories_[0]:
                count += 1
        if count > 0:
            return True  # 有异常返回
        else:
            return False  # 无异常返回

    def _addr2int(self, addr):
        # 将点分十进制IP地址转换成十进制整数
        items = [int(x) if self._is_number(x) else 'F' for x in addr.split('.')]
        if items[0] == 'F':
            return 65535  # ipv6地址的最大ip可用数
            # return False
        return self._sum_of_list(items, len(items))

    def _sum_of_list(self, lists, size):
        if size == 0:
            return 0
        else:
            return lists[size - 1] + self._sum_of_list(lists, size - 1)

    @staticmethod
    def _is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    @staticmethod
    def _porttype_encode(port):
        if 0 < port < 1023:
            port_dynamic = 1
            port_static = 0
        elif 1024 < port < 65535:
            port_dynamic = 0
            port_static = 1
        else:
            port_dynamic = 0
            port_static = 0
        return port_dynamic, port_static

    def data_processing(self, deal_jsonlist, path_featuresincsv):
        """
            对进行初步数据清洗后的数据进行标准化处理（含离散数据编码）
            :param deal_jsonlist:处理后的json字典列表
            :param path_featuresincsv:本地csv文件地址
            :return: flow2df:字面意思
        """

        # ----------
        #  读取数据
        # ----------
        flow2df = pd.DataFrame(deal_jsonlist)
        # scalerdata2df = pd.DataFrame(datalist4scaler)
        # 读取特征字段
        df_col = pd.read_csv(path_featuresincsv, encoding='ISO-8859-1')

        # 小写列名并删除空格
        df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())
        flow2df.columns = df_col['Name']
        # scalerdata2df.columns = df_col['Name']

        del df_col
        # 填充'service'字段的空值.apply(lambda x: x.strip().lower())
        flow2df['service'] = flow2df.service.fillna(value='unknown').apply(str).apply(lambda x: x.strip().lower())
        # flow2df['service'] = flow2df.service.apply(lambda x: x.strip().lower())
        # scalerdata2df['service'] = scalerdata2df.service.fillna(value='unknown').apply(lambda x: x.strip().lower())
        # flow2df[['local_orig', 'local_resp']] = (flow2df[['local_orig', 'local_resp']] == 'True').astype(int)
        # print(flow2df['local_orig'].dtypes)
        # print(flow2df['local_resp'].dtypes)
        flow2df['local_orig'] = flow2df['local_orig'].astype(int)
        flow2df['local_resp'] = flow2df['local_orig'].astype(int)
        # flow2df[['local_orig', 'local_resp']].astype(float)


        # 同理，处理其他含空值的字段（没有做转换后的数值型规定）
        all_col = list(flow2df.columns)
        for feature in all_col:
            if feature not in ['id.orig_h', 'id.resp_p', 'id.resp_h', 'proto', 'conn_state', 'service']:
                flow2df[feature] = flow2df[feature].apply(pd.to_numeric, errors='coerce').fillna(
                    flow2df[feature].mean())  # 均值填充
                # scalerdata2df[feature] = scalerdata2df[feature].apply(pd.to_numeric, errors='coerce').fillna(
                #     scalerdata2df[feature].mean())
                print(f"{feature}列处理完成")
        # print(all_col)
        return flow2df

    def feature_engineering(self, flow2df, ohe_service, ohe_proto, ohe_connstate):
        """
        对进行初步数据清洗后的数据进行标准化处理（含离散数据编码）
        :param flow2df: dataframe格式的日志流数据
        :param service_path: 生成训练集时使用的onehot编码器
        :param proto__path: 生成训练集时使用的onehot编码器
        :param connstate_path: 生成训练集时使用的onehot编码器
        :return: all_data_minmaxpd: 标准化后的数据，可用于模型的训练或预测
        """

        # 因为源IP、源端口、目的IP已经用于计算访问关系
        # 这三个字段的信息已经包含在access_rsid中，所以在标准化时可以舍弃orig_ip_bytes
        flow2df.drop(['id.orig_h', 'id.resp_p', 'id.resp_h'], axis=1, inplace=True)
        # ----------
        #  标准化
        # ----------
        # 无序多分类变量编码
        '''字符类型数据标准化（使用one-hot by sklearn）'''
        # 对给定的列进行one-hot编码并删除原有的列

        for col, ohe in zip(['proto', 'service', 'conn_state'], [ohe_proto, ohe_service, ohe_connstate]):
            x = ohe.transform(flow2df[col].values.reshape(-1, 1))
            tmp_df = pd.DataFrame(x.todense(), columns=[col + '_' + i for i in ohe.categories_[0]])
            flow2df = pd.concat([flow2df.drop(col, axis=1), tmp_df], axis=1).reindex(tmp_df.index)
        # print(flow2df.head()),DataFrame:(345,35)↓
        '''统一数值长度'''
        flow2df.apply(pd.to_numeric, errors='coerce')

        flow2df_rownamelist = flow2df.columns.values
        low4scaling, upper4scaling = 0, 1
        for rowname in flow2df_rownamelist:
        # ## 仅做数值缩放
            # if rowname != 'missed_bytes':
            # print(flow2df[rowname].dtypes)
            flow2df[rowname] = scaling(low4scaling, upper4scaling, flow2df[rowname], scaling_mode=opt.operating_mode)
            # print(train['duration'])
            # flow2df[rowname] = flow2df[rowname].round(0)  # 四舍五入
        # 在这里改
        if opt.operating_mode == 'detection':
            flow2list = (flow2df.values[-1]).tolist()  # torch.Size([35, ])
            # 需要一个加入“访问行为标签”的函数,ndarray:(36,)
            flow36 = self._flow2matrix(flow2list)
            flow2tensor = torch.tensor(flow36, dtype=torch.float64)  # torch.Size([36, ])
            flowdata = flow2tensor.view(1, 36)  # 转变tensor形状：torch.Size([36, ])-->torch.Size([1, 36])

            return flowdata  # 若用作预测就仅返回最后（最新的）一条，其余数据用作缩放参考,DataFrame:(345,35)

        elif opt.operating_mode == 'train':
            # data_path = r"D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\datalist_from_es.csv"
            # flow2df.to_csv(data_path, index=False)
            flow2list = flow2df.values.tolist()
            print(flow2list)
            flow36 = self._flow2matrix(flow2list)

            return flow36

        # # 软编码
        # '''布尔类型数据标准化(软编码，转化为0(0.1~0.4)/1(0.6~0.9)的数据)'''
        # for bcol in binary_col:
        #     print(f"正在对{bcol}列进行软编码")
        #     all_data[bcol] = all_data[bcol].replace([True], np.random.uniform(0.6, 0.9))
        #     all_data[bcol] = all_data[bcol].replace([False], np.random.uniform(0.1, 0.4))
        #     all_data[bcol] = all_data[bcol].apply(pd.to_numeric, errors='coerce').fillna(all_data[bcol].mean())

        # # 数据划分为训练/测试集
        # train_data, test_data = train_test_split(all_data_minmaxpd, test_size=0.3, random_state=16)

    def _flow2matrix(self, data_list):
        """
        用于在flow中添加访问关系的对应标签
        """
        flow_id_inquire = []  # 用于查询访问关系
        if opt.operating_mode == 'detection':
            if data_list[-2] not in flow_id_inquire:
                # 为网络流的访问行为添加对应标签
                flow_id_inquire.append(data_list[-2])
                data_list.append(flow_id_inquire.index(data_list[-2]) + 1)
            else:
                data_list.append(flow_id_inquire.index(data_list[-2]) + 1)

            fa = numpy.array(data_list)
            return fa
        elif opt.operating_mode == 'train':
            data_added = []
            for i, row in enumerate(data_list):
                if row[-2] not in flow_id_inquire:
                    # 为网络流的访问行为添加对应标签
                    # 注意access_RSID在csv中的位置
                    flow_id_inquire.append(row[-2])
                    row.append(flow_id_inquire.index(row[-2])+1)
                else:
                    row.append(flow_id_inquire.index(row[-2]) + 1)
                # flow_list.append(row)
                # print(row)1111
                # processed_row = [int(float(x)) for x in row]
                fa = numpy.array(row)
                data_added.append(fa)
        return data_added







    @staticmethod
    def _element_coding(flowarray, codemode):
        '''
        转换array内元素的编码类型
        :param flowarray: 一个np array，一条flow
        :param codemode: 编码(0) or 解码(1)
        :return:
        '''
        arrayback = []
        for flow in flowarray:
            if codemode == 0:
                arrayback.append(flow.encode())
            elif codemode == 1:
                arrayback.append(flow.decode())

        return arrayback
