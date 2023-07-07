import os
import h5py
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import TensorDataset

class InputData(object):
    """A single set of data."""

    def __init__(self, flow, accessidlabel):
        """
        :param flow:       标准化后的网络流矩阵
        :param accessidlabel:  网络流访问关系id
        """
        self.flow = flow
        self.accessidlabel = accessidlabel


class Dataset4netflow(Dataset):
    def __init__(self):
        self.mode = 'train'

    def get_dataset(self, rawdata):

        if self.mode == 'detection':
            rawdata_path = rawdata
            flowset = self.hdf5_loader(rawdata_path)
        elif self.mode == 'train':
            flowset = self.estraindata_loader(rawdata)

        #不能用torch.long，因为会下取整导致数据丢失
        all_flow = torch.tensor([f.flow for f in flowset], dtype=torch.float64)
        all_flowlabel = torch.tensor([f.accessidlabel for f in flowset], dtype=torch.float64)


        data = TensorDataset(all_flow, all_flowlabel)

        return data

    @staticmethod
    def hdf5_loader(filepath):
        # 返回一个list
        # [[flow1], [flow2],..., [flow n]]
        # 供init分割标签和数据
        # getitem直接返回数据
        flowset = []
        print("=====Loading Data=====")
        f = h5py.File(filepath, "r")
        for i in tqdm(f, desc="Loading hdf5 data and convert to nparray"):
            # 读取hdf5数据，转换为nparray
            # 矩阵内元素类型为binary
            array4ascii = np.array(f[i])
            # npflow = array4ascii.astype('U13')
            # print(i)
            # 将binary转为float,list
            flow2float = [float(x) for x in array4ascii.tolist()]
            # del flow2float[-1]  # 访问关系，不删除也行，将其作为一个特征比较合理
            flowset.append(InputData(flow=flow2float, accessidlabel=int(float(eval(array4ascii[-2])))))
            # print(flow)
        return flowset

    @staticmethod
    def estraindata_loader(flowlistdata):
        flowset = []
        print("=====Loading Data=====")
        for i in tqdm(flowlistdata, desc="Loading hdf5 data and convert to nparray"):
            flow2float = i.tolist()
            flowset.append(InputData(flow=flow2float, accessidlabel=int(flow2float[-2])))
        return flowset


# ----------
#  Test
# ----------
if __name__ == '__main__':

    filepath = r"D:\code\log4gan_dataset\dataset_h5\220922_flowdata_h5py.hdf5"
    device = torch.device('cuda')

    dataprocess = Dataset4netflow()
    train_data = dataprocess.get_dataset(filepath)
    train_data_loader = DataLoader(train_data, batch_size=64,
                                   shuffle=False, drop_last=True)

    for ep in trange(20, desc="Epoch"):
        for step, batch in enumerate(tqdm(train_data_loader, desc="DataLoader")):
            batch = tuple(t.to(device) for t in batch)
            flow, _ = batch
            print(flow)



