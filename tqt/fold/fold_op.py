import torch
import torch.nn as nn
from collections import OrderedDict
from .convbnact import Conv2dBNReLU


def fold_op(conv, bn, relu):
    folded = Conv2dBNReLU(conv, bn, relu)
    conv = folded
    bn = nn.Identity()
    relu = nn.Identity()
    return conv, bn, relu


def fold_the_network(net):
    key = list(net._modules.keys())
    keylen = len(key)
    flag = 0
    while flag < keylen:
        if list(net._modules[key[flag]]._modules.keys()) != []:
            fold_the_network(net._modules[key[flag]])
        else:
            if isinstance(net._modules[key[flag]],
                          nn.Conv2d) and flag + 1 < keylen:
                if isinstance(net._modules[key[flag + 1]],
                              nn.BatchNorm2d) and flag + 2 < keylen:
                    if isinstance(net._modules[key[flag + 2]],
                                  nn.ReLU) or isinstance(
                                      net._modules[key + 2], nn.ReLU6):
                        net._modules[key[flag]], net._modules[key[
                            flag + 1]], net._modules[key[flag + 2]] = fold_op(
                                net._modules[key[flag]],
                                net._modules[key[flag + 1]],
                                net._modules[key[flag + 2]])
                        print(flag)
                        flag += 2
        flag += 1
