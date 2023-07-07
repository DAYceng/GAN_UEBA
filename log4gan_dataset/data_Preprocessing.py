import pandas as pd  # for csv files and dataframe
import warnings
warnings.filterwarnings("ignore")





# ----------
#  读取数据
# ----------
saved_dict = {} #用于保存转换测试数据时使用到的参数
# 读取特征字段
df_col = pd.read_csv(r'D:\code\log4gan_dataset\zeeklog\22.9.19\conn_features9.19.csv', encoding='ISO-8859-1')

# 小写列名并删除空格
df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())

# 读取原始数据文件（csv）
dfs = []
path = r'D:\code\log4gan_dataset\zeeklog\22.9.19\conn4dataset9.19.csv'  # conn.log的csv格式文件
dfs.append(pd.read_csv(path.format(1), header=None))
# 将读取到的单个dataframe整合起来
all_data = pd.concat(dfs).reset_index(drop=True)


all_data.columns = df_col['Name']# 用正确的列名重命名dataframe
saved_dict['columns'] = df_col['Name'][df_col['Name'] != 'label'].tolist()# 保存参数
del df_col
# print(all_data.shape)
# print(all_data.head())

# # 数据划分为训练/测试集
# train, test = train_test_split(all_data, test_size=0.3, random_state=16)
# del all_data
# # print(train.shape, '\n', test.shape)
# # train = all_data  # 或者用的时候自己划分，这里不划分了



# ----------
#  空值处理
# ----------
# print(train.isnull().sum())# 检查空值，用合适的值将其填充
# 字段内容统计
# print(train['service'].value_counts())
# 对含有空值的项进行填充处理
# for column in list(train.columns[train.isnull().sum() > 0]):
#     print(column)
#     mean_val = train[column].mean()
#     train[column].fillna(mean_val, inplace=True)all_data
all_data['service'] = all_data.service.fillna(value='unknown').apply(lambda x: x.strip().lower())
# train['service'] = train.service.fillna(value='unknown').apply(lambda x: x.strip().lower())
# test['service'] = test.service.fillna(value='unknown').apply(lambda x: x.strip().lower())

# 同理，处理其他含空值的字段（没有做转换后的数值型规定）
all_col = list(all_data.columns)
# all_col = list(train.columns)
for feature in all_col:
    if feature not in ['id.orig_h', 'id.resp_p', 'id.resp_h',  'proto', 'conn_state', 'service']:
        all_data[feature] = all_data[feature].apply(pd.to_numeric, errors='coerce').fillna(all_data[feature].mean())  # 均值填充
        # train[feature] = train[feature].apply(pd.to_numeric, errors='coerce').fillna(train[feature].mean())  # 均值填充
        # test[feature] = test[feature].apply(pd.to_numeric, errors='coerce').fillna(test[feature].mean())
        print(f"{feature}列处理完成")
print(all_col)
# print(train.isnull().sum())# 检查空值


# ----------
#  保存数据
# ----------
print("正在保存训练集与测试集数据")
all_data.to_csv(r"D:\code\log4gan_dataset\zeeklog\22.9.28\alldata_EDA.csv", index=False)
# train.to_csv(r"D:\code\log4gan_dataset\zeeklog\22.9.28\train_alldata_EDA.csv", index=False)
# test.to_csv(r"D:\code\log4gan_dataset\zeeklog\22.9.28\test_alldata_EDA.csv", index=False)
