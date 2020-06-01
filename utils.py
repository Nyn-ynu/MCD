import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import params
import torch.nn as nn


# 获取数据集
def get_data(data_name, transform, is_train=True):
    if data_name =="mnist":
        data_loader = datasets.MNIST(root=params.dataset_root, train=is_train, download=True, transform=transform)
    elif data_name == "usps":
        data_loader = datasets.USPS(root=params.dataset_root, train=is_train, download=True, transform=transform)
    data = DataLoader(dataset=data_loader, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    return data


# 初始化神经网络权重
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.orthogonal_(m.weight)
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)


# 损失函数
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1)- F.softmax(out2)))

