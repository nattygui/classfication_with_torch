# -*- coding:utf-8 -*-
class BaseConfig(object):
    data_root = r'I:\dataset\AID\crop_AID_dataset'  # 数据集的根目录
    model = 'resnet50'

    num_classes = 3  # 分类数
    num_epochs = 100
    batch_size = 32
    lr = 0.01  # 初始lr
    width = 224
    height = 224

    iter_smooth = 100  # 打印&记录log的频率

config = BaseConfig()