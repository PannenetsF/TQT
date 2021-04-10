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
    mdic = OrderedDict(net.named_children())
    key = list(mdic.keys())
    keylen = len(key)
    flag = 0
    while flag < keylen:
        if list(OrderedDict(mdic[key[flag]].named_children()).keys()) != []:
            fold_the_network(mdic[key[flag]])
        else:
            if isinstance(mdic[key[flag]], nn.Conv2d) and flag + 1 < keylen:
                if isinstance(mdic[key[flag + 1]],
                              nn.BatchNorm2d) and flag + 2 < keylen:
                    if isinstance(mdic[key[flag + 2]], nn.ReLU) or isinstance(
                            mdic[key + 2], nn.ReLU6):
                        mdic[key[flag]], mdic[key[flag + 1]], mdic[key[
                            flag + 2]] = fold_op(mdic[key[flag]],
                                                 mdic[key[flag + 1]],
                                                 mdic[key[flag + 2]])
                        print(flag)
                        flag += 2
        flag += 1
