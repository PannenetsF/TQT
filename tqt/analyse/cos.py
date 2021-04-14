import torch
from ..function import qsigned, qunsigned
from collections import OrderedDict


def cosine_similarity(net, qnet, wba=[8, 16, 8], show=False):
    ret = []
    n_param, q_param = OrderedDict(net.named_parameters()), OrderedDict(
        qnet.named_parameters())
    for name in n_param.keys():
        if name.find('log') == -1:
            qparam = q_param[name]
            if name.find('weight') != -1:
                width = wba[0]
            elif name.find('bias'):
                width = wba[1]
            else:
                width = wba[2]
            param = n_param[name]
            qparam = qsigned(qparam, q_param[name + '_log2_t'], width)
            cos = param.flatten().dot(
                qparam.flatten()) / param.norm() / qparam.norm()
            ret.append((name, cos))
            if show:
                print(f'layer {name} get cosine of {cos}')
