import torch
import torch.nn as nn
import numpy as np
from decimal import Decimal
# m = nn.Conv1d(16, 33, 3, stride=2)
# input = torch.randn(20, 16, 50)
# output = m(input)


# N = 64
# C_in = 1
# L_in = 7
# inputs = torch.rand([N, C_in, L_in])
# padding = 1
# kernel_size = 5
# stride = 2
# C_out = 1
# x0 = torch.nn.Upsample(scale_factor=2, mode='nearest')
# x1 = torch.nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding)
# x2 = torch.nn.BatchNorm1d(1)# nn.MaxPool1d
# # x2 = torch.nn.MaxPool1d(3, stride=1)
# y0 = x0(inputs)
# print(y0.shape)
# y1 = x1(y0)
# y2 = x2(y1)
# print(y2)
# print(y2.shape)

# import math
# inputs = 4
# # sumlist = []
# sum = 0
# for i in range(1, inputs+1):
#     # print(i)
#     a = math.log(i, 2)+1
#     sum = sum + int(a)
# print(sum)

# inputs = input()
# count = 0
# x_next = 0
# count4two = 0
# while(x_next != 1):
#     if x_next % 2 == 0:
#         x_next = int(inputs)/2
#         count += 1
#         count4two += 1
#     elif int(inputs) % 2 != 0:
#         x_next = int(inputs) - 1
#         count += 1
# if count4two :
#     print(count+count4two)
# else:
#     print(count)
# array4ascii = np.array([b'3.4409539796e-07'])
# npflow = array4ascii.astype("float32")
#
# print(npflow)
# print(eval(b'3.4409539796e-07'))
# print(float(b'3.4409539796e-07'))

# #根据给定的tensor的形状
# t=torch.Tensor([[1,2,3],[4,5,6]])
# #查看tensor的形状
# print(t.size())
# #shape与size()等价方式
# print(t.shape)
# #根据已有形状创建tensor
# cc=torch.Tensor(t.size())
# print(cc)
# tt = torch.randn(t.size())
# print(tt)

# import csv
# toCSV = [{'name':'bob','age':25,'weight':200},
#          {'name':'jim','age':31,'weight':180}]
# with open('people.csv', 'w', encoding='utf8', newline='') as output_file:
#     fc = csv.DictWriter(output_file,
#                         fieldnames=['name', 'age', 'weight'])
#     fc.writeheader()
#     fc.writerows(toCSV)

# testlist = ['OTH' 'REJ' 'RSTO' 'RSTR' 'RSTRH' 'S0' 'S1' 'S2' 'S3' 'SF' 'SHR']
# nornmallist = ['dns', 'udp', 'OTH']
# assholelist = ['dns', 'udp', 'CC']
# # ziduan = 'CC'
# # if ziduan not in testlist:
# #     print(1)
# from gan4netflow.implementations.datapreprocessing import Preprocessing
# from elasticsearch import Elasticsearch
# import pickle
# test_Preprocessing = Preprocessing()
#
# # 获取onehot编码器
# ohe_service = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_service.pkl', 'rb'))
# ohe_proto = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_proto.pkl', 'rb'))
# ohe_connstate = pickle.load(open(r'D:\zpg\paper2\code\UEBA_GAN\gan4netflow\implementations\wgan_gp\onehot\ohe_ohe_connstate.pkl', 'rb'))
#
#
# print(test_Preprocessing._check_odd(nornmallist, ohe_service, ohe_proto, ohe_connstate))

import torch

testrealsample_tensor = torch.tensor([[9.2355e-09, 3.4912e-07, 1.0413e-07, 1.0000e+00, 0.0000e+00, 1.8894e-05,
         2.6675e-06, 2.0270e-05, 4.3418e-06, 1.0000e+00, 0.0000e+00, 1.0895e-02,
         0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00]],
       dtype=torch.float64)

