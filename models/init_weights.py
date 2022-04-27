'''
# Author: yangbinchao
# Date:   2021-12-11
# Email:  heroybc@qq.com
# Describe: 常见的几种初始化方式进行网络参数初始化、参数打印等
'''


import torch
import torch.nn as nn
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__  # 得到网络层的名字
    if classname.find('Conv') != -1:  # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    '''
    选择参数初始化方式
    '''
    if init_type == 'normal':
        net.apply(weights_init_normal)  # apply(fn)：将fn函数递归地应用到网络模型的每个子模型中，主要用在参数的初始化。
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def print_model_params(net):
    '''
    查看各层的weight和bias
    '''
    params = net.state_dict()
    for k, v in params.items():
        print(k)  # 打印网络中的变量名，找出自己想查看的参数名字，这个与自己定义网络时起的名字有关。
    print(params['net.convt1.weight'])   # 打印convt1的weight
    print(params['net.convt1.bias']) # 打印convt1的bias