import torch
from ..function import qsigned, qunsigned
from ..threshold import add_hook, hook_handler, get_hook


def kl_divergence(ha, hb):
    r'''
        ans = J_{kl}(a,b)
    '''
    return (ha * (ha / hb).log()).sum()


def kl_similarity(net, qnet, cali_batch, show=False):
    ret = []
    add_hook(net, 'net', hook_handler, show=False)
    add_hook(qnet, 'net', hook_handler, show=False)
    net(cali_batch)
    qnet(cali_batch)
    n_hook = get_hook(net, '', show=False)
    q_hook = get_hook(qnet, '', show=False)
    for n, q in zip(n_hook, q_hook):
        kl = kl_divergence(n[1].flatten(), q[1].flatten())
        ret.append((n[0], kl))
        if show:
            print(f'kl at {n[0]} is {kl}, average is {kl/n[1].numel()}')

    return ret