from .max import *
from .sd import *
from .kl import *


def init_weight(net_module, qnet_module, method='max'):
    if method == 'max':
        threshold_weight_max(net_module, qnet_module)
    elif method == '3sd':
        threshold_weight_3sd(net_module, qnet_module)
    else:
        raise NotImplementedError()


def init_bias(net_module, qnet_module, method='max'):
    if method == 'max':
        threshold_bias_max(net_module, qnet_module)
    elif method == '3sd':
        threshold_bias_3sd(net_module, qnet_module)
    else:
        raise NotImplementedError()


def init_acti(net_module, qnet_module, bin_number=2048, cali_number=128):
    entropy_calibration(net_module,
                        qnet_module,
                        bin_number=bin_number,
                        cali_number=cali_number)


def init_network(net_proc,
                 qnet_proc,
                 weight_method='max',
                 bias_method='max',
                 bin_number=2048,
                 cali_number=128,
                 show=False):
    if isinstance(net_proc, nn.ModuleList):
        for (module, qmodule) in zip(net_proc, qnet_proc):
            init_network(module,
                         qmodule,
                         weight_method=weight_method,
                         bias_method=bias_method,
                         show=show)
    elif isinstance(net_proc, nn.Sequential):
        for (child, qchild) in zip(net_proc.children(), qnet_proc.children()):
            init_network(child,
                         qchild,
                         weight_method=weight_method,
                         bias_method=bias_method,
                         show=show)
    elif isinstance(net_proc, nn.ModuleDict):
        for (key, qkey) in zip(net_proc.keys(), qnet_proc.keys()):
            init_network(net_proc[key],
                         qnet_proc[qkey],
                         weight_method=weight_method,
                         bias_method=bias_method,
                         show=show)
    else:  # nn.Module()
        if hasattr(qnet_proc, 'acti_log2_t'):
            init_acti(net_proc, qnet_proc, bin_number=2048, cali_number=128)
            if show:
                print(qnet_proc, ': activation threshold is quanted')
        if hasattr(qnet_proc, 'weight_log2_t'):
            init_weight(net_proc, qnet_proc, method=weight_method)
            if show:
                print(qnet_proc, ': weight threshold is quanted')
        if hasattr(qnet_proc, 'bias_log2_t'):
            init_bias(net_proc, qnet_proc, method=bias_method)
            if show:
                print(qnet_proc, ': bias threshold is quanted')
