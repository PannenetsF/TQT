import torch
from ..function import qsigned, qunsigned


def cosine_similarity(net, qnet, show=False):
    ret = []
    n_param, q_param = dict(net.named_parameters()), dict(
        qnet.named_parameters())
    for (name, param), (_, qparam) in zip(n_param, q_param):
        if name.find('log') == -1:
            qparam = qsigned(qparam, q_param[name + '_log2_t'],
                             q_param[name + '_bit_width'])
            cos = param.flatten().dot(
                qparam.flatten()) / param.norm() / qparam.norm()
            ret.append((name, cos))
        if show:
            print(f'layer {name} get cosine of {cos}')
