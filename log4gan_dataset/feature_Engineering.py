import pandas as pd  # for csv files and dataframe
import random
import numpy as np
import warnings

from sklearn import preprocessing

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 载入训练/测试数据
all_data = pd.read_csv(r"D:\code\log4gan_dataset\zeeklog\22.9.28\alldata_EDA.csv")
# train = pd.read_csv(r"D:\code\log4gan_dataset\zeeklog\22.9.28\train_alldata_EDA.csv")
# test = pd.read_csv(r"D:\code\log4gan_dataset\zeeklog\22.9.28\test_alldata_EDA.csv")


# test = pd.read_csv('./test_alldata_EDA.csv')

# proto_inquire_list = []
# service_inquire_list = []
# connstate_inquire_list = []

def mad(dataframe_row):
    '''
    异常值处理
    方法：绝对中位差,median absolute deviation
    :param dataframe_row:目标dataframe中待处理的列
    :return: 处理完毕指定列后的dataframe
    '''
    x = dataframe_row.median()  # 求列数据的中位数
    subtract = dataframe_row - x  # 每个值都减去x
    subtract = abs(subtract)  # 列数据取绝对值
    mc = subtract.median()  # 求绝对值化后的列中位数
    mad = mc * 1.4826  # 将mc转换为绝对中位差mad
    # 求异常值范围
    # 使用初始列的中位数加减MAD的倍数，倍数一般取2.5
    upper_limit = x + 2.5 * mad
    lower_limit = x - 2.5 * mad

    return upper_limit, lower_limit


def scaling(low4scaling, upper4scaling, dataframe_row):
    '''
    将给定的df列数据缩放都指定范围
    :param uplimit: 缩放范围上限
    :param donwlimit: 缩放范围下限
    :param dataframe_row: 目标dataframe中待处理的列
    :return:
    '''
    # 找出列中的最值
    row_max = dataframe_row.max()
    row_min = dataframe_row.min()
    dataframe_row = (dataframe_row - row_min) / (row_max - row_min) * (
            upper4scaling - low4scaling) + low4scaling
    return dataframe_row


if __name__ == '__main__':
    # 因为源IP、源端口、目的IP已经用于计算访问关系
    # 这三个字段的信息已经包含在access_rsid中，所以在标准化时可以舍弃orig_ip_bytes
    all_data.drop(['id.orig_h', 'id.resp_p', 'id.resp_h'], axis=1, inplace=True)
    # train.drop(['id.orig_h', 'id.resp_p', 'id.resp_h'], axis=1, inplace=True)
    # test.drop(['id.orig_h', 'id.resp_p', 'id.resp_h'], axis=1, inplace=True)

    print(all_data.columns)
    # 区分出数值类的列
    cat_col = ['proto', 'service', 'conn_state']  # 策略类型
    binary_col = ['local_orig', 'local_resp']  # 布尔类型
    num_col = list(set(all_data.columns) - set(cat_col) - set(binary_col))  # 数值类型


    # ----------
    #  标准化
    # ----------

    # 无序多分类变量编码

    '''字符类型数据标准化（使用one-hot by sklearn）'''
    service_ = OneHotEncoder()
    proto_ = OneHotEncoder()
    conn_state_ = OneHotEncoder()
    ohe_service = service_.fit(all_data.service.values.reshape(-1, 1))
    ohe_proto = proto_.fit(all_data.proto.values.reshape(-1, 1))
    ohe_connstate = conn_state_.fit(all_data.conn_state.values.reshape(-1, 1))

    # ohe_service4train = service_.fit(train.service.values.reshape(-1, 1))
    # ohe_proto4train = proto_.fit(train.proto.values.reshape(-1, 1))
    # ohe_connstate4train = conn_state_.fit(train.conn_state.values.reshape(-1, 1))

    # 对给定的列进行one-hot编码并删除原有的列
    for col, ohe in zip(['proto', 'service', 'conn_state'], [ohe_proto, ohe_service, ohe_connstate]):
        x = ohe.transform(all_data[col].values.reshape(-1, 1))
        tmp_df = pd.DataFrame(x.todense(), columns=[col + '_' + i for i in ohe.categories_[0]])
        all_data = pd.concat([all_data.drop(col, axis=1), tmp_df], axis=1).reindex(tmp_df.index)
    print(all_data.head())


    # for col, ohe in zip(['proto', 'service', 'conn_state'], [ohe_proto4train, ohe_service4train, ohe_connstate4train]):
    #     x = ohe.transform(train[col].values.reshape(-1, 1))
    #     tmp_df = pd.DataFrame(x.todense(), columns=[col + '_' + i for i in ohe.categories_[0]])
    #     train = pd.concat([train.drop(col, axis=1), tmp_df], axis=1).reindex(tmp_df.index)
    # # del train['service_krb']
    # # del train['service_radius']  # 会莫名多出这两个字段，删除之
    # print(train.head())

    # ohe_service4test = service_.fit(test.service.values.reshape(-1, 1))
    # ohe_proto4test = proto_.fit(test.proto.values.reshape(-1, 1))
    # ohe_connstate4test = conn_state_.fit(test.conn_state.values.reshape(-1, 1))

    # for col, ohe in zip(['proto', 'service', 'conn_state'], [ohe_proto4test, ohe_service4test, ohe_connstate4test]):
    #     x = ohe.transform(test[col].values.reshape(-1, 1))
    #     tmp_df = pd.DataFrame(x.todense(), columns=[col + '_' + i for i in ohe.categories_[0]])
    #     test = pd.concat([test.drop(col, axis=1), tmp_df], axis=1).reindex(tmp_df.index)

    # print(test.head())

    # '''非数值类型数据标准化（使用直接映射）'''
    # for col in ['proto', 'service', 'conn_state']:
    #     if col == 'proto':
    #         x = train[col]
    #         proto_inquire_chart = x.unique()  # 获取无重复的"proto"数据列
    #         for i, type in enumerate(proto_inquire_chart):
    #             train.loc[train[col].isin([type]), 'proto'] = i + 25  # 将proto数据列中的数据替换为其在proto_inquire_chart中的对应索引
    #         # print(train[col])
    #         train[col] = train[col].apply(pd.to_numeric, errors='coerce')
    #     elif col == 'service':
    #         x = train[col]
    #         service_inquire_chart = x.unique()
    #         for i, type in enumerate(service_inquire_chart):
    #             # 在当前下标的基础上，加上一个0~255内的随机数(25)作为偏移
    #             # 避免在标准化之后出现相同的数值
    #             train.loc[train[col].isin([type]), 'service'] = i + 25 + len(service_inquire_chart)
    #         # print(train[col])
    #         train[col] = train[col].apply(pd.to_numeric, errors='coerce')
    #     elif col == 'conn_state':
    #         x = train[col]
    #         cs_inquire_chart = x.unique()
    #         for i, type in enumerate(cs_inquire_chart):
    #             train.loc[train[col].isin([type]), 'conn_state'] = i + 25 + len(cs_inquire_chart)
    #         # print(train[col])
    #         train[col] = train[col].apply(pd.to_numeric, errors='coerce')

    # '''非数值类型数据标准化（使用虚拟变量编码）'''
    # for col in ['proto', 'service', 'conn_state']:
    #     if col == 'proto':
    #         x = train[col]
    #         print(x.head())
    #         train = pd.get_dummies(x)
    #         # train[col] = pd.get_dummies(train[col], drop_first=True)
    #         print(train[col])
    #     elif col == 'service':
    #         x = train[col]
    #         pass
    #     elif col == 'conn_state':
    #         x = train[col]
    #         pass


    '''统一数值长度'''
    all_data.apply(pd.to_numeric, errors='coerce')
    # train.apply(pd.to_numeric, errors='coerce')
    # test.apply(pd.to_numeric, errors='coerce')
    # # print(train.dtypes())
    # # scaler = StandardScaler()
    # # scaler = scaler.fit((train[all_col]))
    # # 数值类型数据标准化
    # # 标准化会将布尔类型数据转换为0/1表示
    all_col = list(all_data.columns)
    # all_col.sort(key=train.columns)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = min_max_scaler.fit((all_data[all_col]))
    all_data_minmax = min_max_scaler.transform(all_data[all_col])
    all_data_minmaxpd = pd.DataFrame(all_data_minmax, columns=all_col)

    # # 软编码
    # '''布尔类型数据标准化(软编码，转化为0(0.1~0.4)/1(0.6~0.9)的数据)'''
    # for bcol in binary_col:
    #     print(f"正在对{bcol}列进行软编码")
    #     all_data[bcol] = all_data[bcol].replace([True], np.random.uniform(0.6, 0.9))
    #     all_data[bcol] = all_data[bcol].replace([False], np.random.uniform(0.1, 0.4))
    #     all_data[bcol] = all_data[bcol].apply(pd.to_numeric, errors='coerce').fillna(all_data[bcol].mean())

    # 数据划分为训练/测试集
    train_data, test_data = train_test_split(all_data_minmaxpd, test_size=0.3, random_state=16)


    # min_max_scaler4train = preprocessing.MinMaxScaler()
    # min_max_scaler4test = preprocessing.MinMaxScaler()
    #
    # min_max_scaler4train = min_max_scaler4train.fit((train[all_col]))
    # train_minmax4train = min_max_scaler4train.transform(train[all_col])
    # train_minmaxpd = pd.DataFrame(train_minmax4train, columns=all_col)
    #
    #
    # min_max_scaler4test = min_max_scaler4test.fit((test[all_col]))
    # test_minmax4test = min_max_scaler4test.transform(test[all_col])
    # test_minmaxpd = pd.DataFrame(test_minmax4test, columns=all_col)



    # train_rownamelist = train.columns.values
    # low4scaling, upper4scaling = -1, 1
    # for rowname in train_rownamelist:
    #     # # 去除异常值
    #     # upper_limit, lower_limit = mad(train[rowname])
    #     # if upper_limit != lower_limit:
    #     #     if rowname != 'access_rsid':
    #     #         train = train[(train[rowname] >= lower_limit) & (train[rowname] <= upper_limit)]  # 舍弃含异常值的网络流（行数据）
    #     #     # print(train)
    #     #     ## 计算缩放范围
    #     #     train[rowname] = scaling(low4scaling, upper4scaling, train[rowname])
    #     #     # print(train['duration'])
    #     #     train[rowname] = train[rowname].round(0)  # 四舍五入
    #     #     # print(train[rowname])
    #     # elif upper_limit == lower_limit and train[rowname].max() > 255:
    #     #     train[rowname] = scaling(low4scaling, upper4scaling, train[rowname])
    #     #     # print(train['duration'])
    #     #     train[rowname] = train[rowname].round(0)  # 四舍五入
    #     #     # print(train[rowname])
    #     # else:
    #     #     pass
    #
    #     # ## 仅做数值缩放
    #     # if rowname != 'missed_bytes':
    #     train[rowname] = scaling(low4scaling, upper4scaling, train[rowname])
    #     # print(train['duration'])
    #     train[rowname] = train[rowname].round(0)  # 四舍五入

    # ----------
    #  数据保存
    # # ----------
    # file_path = 'dataset_builder/'
    # # 保存数据编码过程中的builder，下次再用就可以直接调
    # pickle.dump(scaler, open(file_path + 'scaler.pkl', 'wb'))  # Standard scaler
    # # pickle.dump(saved_dict, open(file_path + 'saved_dict.pkl', 'wb'))  # Dictionary with important parameters
    # # pickle.dump(mode_dict, open(file_path + 'mode_dict.pkl', 'wb'))  # Dictionary with most frequent values of columns
    #
    # # 保存Onehot 编码器
    # pickle.dump(ohe_proto, open(file_path + 'ohe_proto.pkl', 'wb'))
    # pickle.dump(ohe_service, open(file_path + 'ohe_service.pkl', 'wb'))
    # pickle.dump(ohe_connstate, open(file_path + 'ohe_connstate.pkl', 'wb'))

    # pickle.dump((train, open(file_path + 'final_train.pkl', 'wb')))
    print(train_data['access_rsid'].value_counts())

    train_data.to_csv(r"D:\code\log4gan_dataset\zeeklog\22.9.28\train_inquire.csv", index=False)
    test_data.to_csv(r"D:\code\log4gan_dataset\zeeklog\22.9.28\test_inquire.csv", index=False)
    # # test.to_csv('./test_alldata_oh.csv', index=False)
