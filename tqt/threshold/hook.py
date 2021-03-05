import torch.nn as nn


def hook_handler(model, input, output):
    model.hook_out = output


def add_hook(modulelist, hook_handler):
    handles = []
    for module in modulelist:
        handles.append(module.register_forward_hook(hook_handler))
    return handles


def remove_hook(handles):
    for handle in handles:
        handle.remove()


def add_hook_general(layer, hook_handler):
    handles = []
    if isinstance(layer, nn.ModuleList):
        handles.extend(add_hook(layer, hook_handler))
    elif isinstance(layer, nn.Sequential):
        for child in layer.children():
            handles.extend(add_hook_general(child, hook_handler))
    elif isinstance(layer, nn.ModuleDict):
        for key in layer.keys():
            handles.extend(add_hook_general(layer[key], hook_handler))
    elif hasattr(layer, 'proc'):
        for key in layer.proc:
            handles.extend(add_hook_general(getattr(layer, key), hook_handler))
    else:  # nn.Module()
        handles.append(layer.register_forward_hook(hook_handler))
    return handles


num_list = [str(i) for i in range(0, 10)]


def _getattr(obj, key):
    stage = key.rsplit('.')
    for i in stage:
        if i[0] in num_list:
            obj = obj[int(i)]
        else:
            obj = getattr(obj, i)
    return obj


def add_hook_fromkeys(net, hook_handler):
    okeys = list(net.state_dict().keys())
    mkeys = [x[:x.rfind('.')] for x in okeys]
    handles = []
    for key in mkeys:
        net_proc = _getattr(net, key)
        handles.append(net_proc.register_forward_hook(hook_handler))
    return handles


def add_hook_fromchild(net, hook_handler):
    handles = []
    for i in net.modules():
        handles.extend(add_hook_general(i, hook_handler))
