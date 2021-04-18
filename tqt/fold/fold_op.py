import torch
import torch.nn as nn
from collections import OrderedDict
from .convbnact import Conv2dBNReLU
from .convbn import Conv2dBN


def fold_CBR(conv, bn, relu):
    folded = Conv2dBNReLU(conv, bn, relu)
    conv = folded
    bn = nn.Identity()
    relu = nn.Identity()
    return conv, bn, relu


def fold_CB(conv, bn):
    folded = Conv2dBN(conv, bn)
    conv = folded
    bn = nn.Identity()
    return conv, bn


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
                                      net._modules[key[flag + 2]], nn.ReLU6):
                        net._modules[key[flag]], net._modules[key[
                            flag + 1]], net._modules[key[flag + 2]] = fold_CBR(
                                net._modules[key[flag]],
                                net._modules[key[flag + 1]],
                                net._modules[key[flag + 2]])
                        flag += 2
                    else:
                        net._modules[key[flag]], net._modules[key[
                            flag + 1]] = fold_CB(net._modules[key[flag]],
                                                 net._modules[key[flag + 1]])
                        flag += 1
        flag += 1


def make_the_shortcut_share(net, show=False):
    keys = list(net._modules.keys())
    if hasattr(net, 'share_path'):
        getattr(net, 'share_path')(show=show)
    for k in keys:
        make_the_shortcut_share(net._modules[k], show=show)
