from collections import namedtuple
import torch.nn as nn
from .utils import _isinstance
from .fold import isfoldmodule

# global config for tqt
Fconfig = namedtuple('f_config', ['weight', 'bias', 'acti'],
                     defaults=[8, 8, 8])


def list_to_fconfig(lst, default=[8, 8, 8]):
    config = []
    for i in lst:
        if i == []:
            config.append(Fconfig(default))
        elif len(i) == len(default):
            config.append(Fconfig(i))
        else:
            raise Exception('the config length is incorrect')


def get_all_layers(net, al=[]):
    '''
    get all layers of (fusedmodule/nn.module) for config 
    '''
    for i in net._modules.keys():
        if list(net._modules[i]._modules) == []:
            if not isinstance(net._modules[i], nn.Identity):
                al.append(net._modules[i])
        elif isfoldmodule(net._modules[i]):
            al.append(net._modules[i])
        else:
            al = get_all_layers(net._modules[i], al)
    return al


def layers_config(net, lst):
    '''
    net is a nn.module with folded modules
    lst is a list of Fconfig, indicating the w/b/a of each layer
    '''
    list_flag = 0
    module_flag = 0
    key_len = len(get_all_layers(net))
    key = list(get_all_layers(net))
    list_len = len(lst)
    
    if list_len != key_len:
        raise Exception('config length not match')
    
    for mod, cfg in zip(key, lst):
        if hasattr(mod, 'weight_bit_width'):
            mod.weight_bit_width = cfg.weight
        if hasattr(mod, 'bias_bit_width'):
            mod.bias_bit_width = cfg.bias
        if hasattr(mod, 'acti_bit_width'):
            mod.acti_bit_width = cfg.acti