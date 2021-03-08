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
                 name,
                 weight_method='max',
                 bias_method='max',
                 bin_number=2048,
                 cali_number=128,
                 show=False):
    modules = list(net_proc._modules.keys())
    if modules == []:
        if hasattr(qnet_proc, 'acti_log2_t'):
            init_acti(net_proc, qnet_proc, bin_number=2048, cali_number=128)
            if show:
                print(name, ': activation threshold is quanted as',
                      float(qnet_proc.acti_log2_t))
        if hasattr(qnet_proc, 'weight_log2_t'):
            init_weight(net_proc, qnet_proc, method=weight_method)
            if show:
                print(name, ': weight threshold is quanted as',
                      float(qnet_proc.weight_log2_t))
        if hasattr(qnet_proc, 'bias_log2_t') and qnet_proc.bias is not None:
            init_bias(net_proc, qnet_proc, method=bias_method)
            if show:
                print(name, ': bias threshold is quanted as',
                      float(qnet_proc.bias_log2_t))
    else:
        for key in modules:
            init_network(net_proc._modules[key],
                         qnet_proc._modules[key],
                         name + '.' + key,
                         weight_method=weight_method,
                         bias_method=bias_method,
                         bin_number=bin_number,
                         cali_number=cali_number,
                         show=show)
