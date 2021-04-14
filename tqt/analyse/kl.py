import torch
import torch.nn as nn
from ..function import qsigned, qunsigned
from ..threshold import add_hook, hook_handler, get_hook


def kl_divergence(ha, hb):
    r'''
        ans = J_{kl}(a,b)
    '''
    return (ha * (ha / hb).log()).sum()


def kl_similarity(
        net,
        qnet,
        cali_batch,
        end_list=[nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d, nn.ReLU6],
        show=True):
    ret = []
    add_hook(net, 'net', hook_handler, end_list=end_list, show=True)
    add_hook(qnet, 'net', hook_handler, end_list=end_list, show=True)
    net(cali_batch)
    qnet(cali_batch)
    n_hook = get_hook(net, '', show=True)
    q_hook = get_hook(qnet, '', show=True)
    print(n_hook)
    for n, q in zip(n_hook, q_hook):
        kl = kl_divergence(n[1].flatten(), q[1].flatten())
        ret.append((n[0], kl))
        if show:
            print(f'kl at {n[0]} is {kl}, average is {kl/n[1].numel()}')
    return ret