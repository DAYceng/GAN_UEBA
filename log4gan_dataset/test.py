import h5py
import pandas as pd
import numpy as np
from flow2Matrix import elementcoding
# file_path = r'D:\code\log4gan_dataset\dataset_h5\220914_flowdata_h5py.hdf5'
# f = h5py.File(file_path, "r")
#
# array4ascii = np.array(f['flow_380908'])
# npflow = array4ascii.astype('U13')
# print(npflow)
# print(list(f.keys()))
# for key in f.keys():
#         # print(f[key].name)
#         # print(f[key].shape)
#         if key=='flow_380908':
#             print(f[key].name)
#             print(f[key].shape)
#             print(f[key])
# #
# #         flow_ascii = f[key][:]
# #         flow_array = elementcoding(flow_ascii, codemode=1)
# #         print(len(flow_array))
# import datetime
#
# t = datetime.datetime.now()
# savetime = t.strftime('%y%m%d')
# print(savetime)
# low4scaling, upper4scaling = 0, 65535
# access_RSID = 65535
# access_rsid255 = access_RSID % (upper4scaling-low4scaling+1)+low4scaling
# print(access_rsid255)
# import numpy as np
#
# print(np.random.uniform(0.1,0.5))


# mailto = ['cc', 'bbbb', 'afa', 'sss', 'bbbb', 'cc', 'shafa']
# addr_to = list(set(mailto))
# addr_to.sort(key=mailto.index)
# print(addr_to)

# 导入os模块
import os

# path定义要获取的文件名称的目录
# path = r"D:\code\log4gan_dataset\zeeklog\22.9.28"

# # os.listdir()方法获取文件夹名字，返回数组
# file_name_list = os.listdir(path)
# for i in file_name_list:
#     spiltname = i.split('_')
#     print(spiltname[0])
#
# # 转为转为字符串
# file_name = str(file_name_list)
#
# # replace替换"["、"]"、" "、"'"
# file_name = file_name.replace("[", "").replace("]", "").replace("'", "").replace(",", "\n").replace(" ", "")

# # 创建并打开文件list.txt
# f = open(path + "\\" + "文件list.txt", "a")
#
# # 将文件下名称写入到"文件list.txt"
# f.write(file_name)


test_data = {"ts": "2022-11-27T11:28:27.223249Z", "uid": "CPHPed1o5fB4ucY6El", "id.orig_h": "172.23.1.131",
             "id.orig_p": 62702, "id.resp_h": "224.0.0.252", "id.resp_p": 5355, "proto": "udp", "service": "dns",
             "duration": 0.10699295997619629, "orig_bytes": 74, "resp_bytes": 0, "conn_state": "S0", "local_orig": 1,
             "local_resp": 0, "missed_bytes": 0, "history": "D", "orig_pkts": 2, "orig_ip_bytes": 130,
             "resp_pkts": 0, "resp_ip_bytes": 0, "application2": "zeek"}


print(pd.from_dict(test_data, orient='index',columns=['fruits']))
