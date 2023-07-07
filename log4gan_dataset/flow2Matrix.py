import h5py
import csv
import numpy
from PIL import Image
import datetime

flow_list = []  # 保存网络流访问关系
flow_id_inquire = []  # 用于查询访问关系

# def save2hdf5(flowarray, whichone, flag, savepath):
#     '''
#     将flow数据保存为hdf5文件
#     :param flowarray: np array形式的flow数据
#     :param whichone: 当前为第几条flow
#     :param flag: 0表示初次写入数据，需要创建hdf5;
#                  1表示再次写入数据，打开现有的hdf5文件
#     :param savepath: 文件保存位置
#     :return:
#     '''
#     array4ascii = []
#     for flow in flowarray:
#         array4ascii.append(flow.encode())
#     if flag == 0:
#         file_mode = "w"
#     elif flag == 1:
#         file_mode = "r"
#
#     sf = h5py.File(savepath + "flowdata_h5py.hdf5", file_mode)
#     datasetname = "flow_" + str(whichone)
#     fd = sf.create_dataset(datasetname, data=array4ascii)
#     for key in sf.keys():
#         print(sf[key].name)
#         print(sf[key][:])
#
#
#     return print(f'保存第{whichone}条数据')


def elementcoding(flowarray, codemode):
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



if __name__ == '__main__':
    datasetlist = ["train_inquire.csv", "test_inquire.csv"]
    for filename in datasetlist:
        print(f"载入{filename}")
        spiltname = filename.split('_')
        datafile_path = f"D:\\code\\log4gan_dataset\\zeeklog\\22.9.28\\{filename}"
        datasave_path = r'D:\code\log4gan_dataset\dataset_h5'

        t = datetime.datetime.now()
        savetime = t.strftime('%y%m%d')
        with open(datafile_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # flag = 0
            for i, row in enumerate(reader):
                if i == 0:
                    continue  # 跳过表头
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
                array4ascii = elementcoding(fa, codemode=0)

                # # if flag == 0:
                # #     file_mode = "w"
                # # elif flag >= 1:
                # #     file_mode = "r"
                # #
                # # sf = h5py.File(datasave_path + "\\flowdata_h5py.hdf5", file_mode)
                # # flag = 1
                # sf = h5py.File(datasave_path + f'./{savetime}_flowdata_h5py.hdf5', 'a')
                # datasetname = "flow_" + str(i)
                # fd = sf.create_dataset(datasetname, data=array4ascii)
                # sf.close()

                # 无需f.close()的写法
                with h5py.File(datasave_path + f'./{savetime}_flowdata_{spiltname[0]}.hdf5', 'a') as sf:
                    datasetname = "flow_" + str(i)
                    fd = sf.create_dataset(datasetname, data=array4ascii)

                print(f'正在保存{spiltname[0]}数据集的第{i}条数据')
                # save2hdf5(flowarray=fa, whichone=i, flag=flag, savepath=datasave_path)
            print('===============保存完成===============')




