# UEBA_GAN说明文档

![image-20221218155212733](https://cdn.jsdelivr.net/gh/DAYceng/drawingBed/blogImg/image-20221218155212733.png)

## 简介
上面有流程图，看不见就挂韡邳蒽，或者去gan4netflow下也有图片文件

查询es中zeek上传的网卡监听日志信息，使用提前训练好的基线模型预测所查询的网络流的行为的得分，进而判断该网络流是否存在异常

项目的主文件夹是：UEBA_GAN/ gan4netflow

### 结构树

----------**gan4netflow** 

-------**data**  # 存放用于本地训练的数据文件（使用时需要自己创建该文件夹）

 ----xxx.hdf5   # 数据文件

--------**implementations**   # 主要文件夹，存放模型代码

...

-----**wgan_gp**  # wgan_gp的实现代码

---**models**  # 存放模型文件（使用时需要自己创建该文件夹）

 ----discriminator_epoch_160.pth   # 模型文件

---**onehot ** # 存放在训练时用于数据标准化的模型的参数

...

 --baseline_demo.py  # 演示demo

 --config.py   # 配置文件（可自行编辑）

 --test(_xxx).py   #  测试各部分功能的测试用例

 --wgan_gp.py  #  wgan_gp代码实现（里面使用了很多测试阶段的本地训练文件，如果你有这些文件，放在对应文件夹下就可以运行了）



本工程主要包括以下部分：

* 查询数据处理
* 数据封装*
* 演示demo

*注：

1、dataprocess4netflow.py为自定义的数据装载器，其继承并重写了DataLoader，主要用于训练模型时，封装和载入自建数据集

​	**截止2022.12.18，本工程尚未添加训练部分的代码，仅支持对已有模型的部署演示**

2、对查询数据的处理主要由datapreprocessing.py中的Preprocessing类实现，详情见注释

3、log4gan_dataset文件夹下是早期用于数据处理的零散代码，其中的大部分逻辑已经重写为Preprocessing类，感兴趣也可以研究研究

## 安装

### 环境配置

根据requirements配置对应的环境

```python
pip install -r requirements.txt
```

### 下载

下载项目到本地

```bash
git clone https://gitee.com/tjpusky/UEBA_GAN.git
```

### 运行测试用例

为了保险起见，请先使用测试用例测试各部分功能是否正常工作

测试用例位于：UEBA_GAN\gan4netflow\implementations\

```
test_get_log_from_es.py
test_check_odd.py
test_extract_Fivetup.py
test_data_Processing.py
test_feature_Engineering.py
test_get_predict_data.py
```

### 运行demo

你可以运行

...\UEBA_GAN\gan4netflow\implementations\wgan_gp\baseline_demo.py

来测试模型是否正常工作
