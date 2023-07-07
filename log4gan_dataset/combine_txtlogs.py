# -*- coding: utf-8 -*-
import os


def read_files():
    """该函数用于读取对应文件夹下各txt文件的名字"""
    path = input("目标文件夹：") + '/'
    # path = r'D:\code\log4gan_dataset\zeeklog\22.8.27\8.26'
    files = os.listdir(path)
    file_names = []
    for file in files:
        if file.split('.')[-1] == 'log':  # 如果不是txt文件就跳过
            file_names.append(file)
    return path, file_names


def mixed_file(path, files):
    """该函数用于合并刚才读取的各文件
    输入：文件路径，read_files()返回的文件名
    输出：一个合并后的文件"""
    content = ''
    for file_name in files:
        with open(path + file_name, 'r', encoding='utf-8') as file:
            content = content + file.read()
            file.close()

    with open(path + 'combine_logs.log', 'a', encoding='utf-8') as file:
        file.write(content)
        content = ''
        file.close()


if __name__ == '__main__':
    path, files = read_files()
    mixed_file(path, files)

