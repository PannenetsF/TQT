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
    else:  # nn.Module()
        handles.append(layer.register_forward_hook(hook_handler))
    return handles
