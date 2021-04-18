from .max import *
from .sd import *
from .kl import *
from .ignore import *
from ..function import qunsigned, qsigned


def init_weight(net_module, qnet_module, method='max'):
    if method == 'max':
        threshold_weight_max(net_module, qnet_module)
    elif method == '3sd':
        threshold_weight_3sd(net_module, qnet_module)
    elif method == 'ig':
        threshold_weight_ig(net_module, qnet_module)
    else:
        raise NotImplementedError()
    qnet_module.weight.data = qsigned(
        qnet_module.weight,
        qnet_module.weight_log2_t.to(qnet_module.weight.device),
        qnet_module.weight_bit_width)


def init_bias(net_module, qnet_module, method='max'):
    if method == 'max':
        threshold_bias_max(net_module, qnet_module)
    elif method == '3sd':
        threshold_bias_3sd(net_module, qnet_module)
    elif method == 'ig':
        threshold_bias_ig(net_module, qnet_module)
    else:
        raise NotImplementedError()
    qnet_module.bias.data = qsigned(
        qnet_module.bias, qnet_module.bias_log2_t.to(qnet_module.bias.device),
        qnet_module.bias_bit_width)


def init_acti(net_module,
              qnet_module,
              method='entro',
              bin_number=2048,
              cali_number=128):
    if method == 'entro':
        entropy_calibration(net_module,
                            qnet_module,
                            bin_number=bin_number,
                            cali_number=cali_number)
    elif method == 'ig':
        threshold_activation_ig(net_module, qnet_module)
    elif method == 'max':
        threshold_activation_max(net_module, qnet_module)
    else:
        raise NotImplementedError()


def init_network(net_proc,
                 qnet_proc,
                 name,
                 weight_method='max',
                 bias_method='max',
                 acti_method='entro',
                 bin_number=2048,
                 cali_number=128,
                 show=False):
    modules = list(net_proc._modules.keys())
    if modules == []:
        if hasattr(qnet_proc, 'acti_log2_t'):
            if qnet_proc.quant is True:
                init_acti(net_proc,
                          qnet_proc,
                          method=acti_method,
                          bin_number=2048,
                          cali_number=128)
                if show:
                    print(name, ': activation threshold is quanted as',
                          float(qnet_proc.acti_log2_t))
        if hasattr(qnet_proc, 'weight_log2_t'):
            if qnet_proc.quant is True:
                init_weight(net_proc, qnet_proc, method=weight_method)
                if show:
                    print(name, ': weight threshold is quanted as',
                          float(qnet_proc.weight_log2_t))
        if hasattr(qnet_proc, 'bias_log2_t') and qnet_proc.bias is not None:
            if qnet_proc.quant is True:
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
                         acti_method=acti_method,
                         bin_number=bin_number,
                         cali_number=cali_number,
                         show=show)
