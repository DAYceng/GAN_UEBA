import binascii
import numpy
import math
import csv
from PIL import Image
import pandas as pd
# train = pd.read_csv(r'D:\code\log4gan_dataset\train_alldata_oh.csv')
PNG_SIZE = 28  # 生成图片大小
flow_str_list = []  # 保存字符串形式网络流
flow_len_list = []  # 保存网络流长度信息
flow_id_list = []  # 保存网络流访问关系
flow_id_inquire = []  # 用于查询访问关系

def getMatrixfrom_pcap(data_bytes, width):
    hexst = binascii.hexlify(data_bytes)
    # 将hexst字符串按三个字节切割后生成对应十六进制的数，并产生一个列表，使用np转换为矩阵
    fh = numpy.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])
    rn = len(fh)/width
    fh = numpy.reshape(fh[:int(rn*width)], (-1, width))
    fh = numpy.uint8(fh)
    return fh

if __name__ == '__main__':
    datafile_path = r'D:\code\USTC-TK2016-master\1_Pcap\tttrain_alldata_oh.csv'

    with open(datafile_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if row[10] not in flow_id_inquire:
                flow_id_inquire.append(row[10])
            flow_id_list.append(row[10])
            flow_len_list.append(len(','.join(row)))
            flow_as_str = ','.join(row)
            flow_str_list.append(flow_as_str)
        # print(flow_id_inquire)
        max_flowlen = max(flow_len_list)
        print("当前最长网络流长度为：", max_flowlen)
        # 上取整，获取flow长度与图片尺寸的公约数
        common_divisor = (math.ceil(max_flowlen/PNG_SIZE))
        # 使用公约数计算填充长度
        padding_len = PNG_SIZE*common_divisor
        print("填充长度：", padding_len)

        # 填充并转换网络流（str）为bytes
        for i, flow_str in enumerate(flow_str_list):
            # print(flow_id_inquire.index(flow_id_list[i]))  # 获取对应id在查表中的位置作为该id的映射编号，即本条网络流的标签
            # 根据flow_str_list的索引，找到查询表中id的下标作为标签
            label = flow_id_inquire.index(flow_id_list[i]) + 1
            flow_str_padding = flow_str.ljust(padding_len, "0")
            # 转换为bytes类型
            flow_as_bytes = str.encode(flow_str_padding)
            fh = getMatrixfrom_pcap(flow_as_bytes, PNG_SIZE)

            im = Image.fromarray(fh)
            im.save(f'./flow2png/connflow_{label}_{i+1}.png')  # 保存生成图片





