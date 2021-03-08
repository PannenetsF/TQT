import torch.nn as nn


def hook_handler(model, input, output):
    model.hook_out = output


def remove_hook(handles):
    for handle in handles:
        handle.remove()


def remove_hookout(layer, name, show=False):
    keys = list(layer._modules.keys())
    if keys == []:
        if show:
            print(name, 'has removed hookout')
        delattr(layer, 'hook_out')
    else:
        for key in keys:
            remove_hookout(layer._modules[key], name + '.' + key, show=show)


def add_hook(layer, name, hook_handler, show=False):
    handles = []
    keys = list(layer._modules.keys())
    if keys == []:
        if show:
            print(name, 'has added a hook')
        handles.append(layer.register_forward_hook(hook_handler))
        return handles
    else:
        for key in keys:
            handles.extend(
                add_hook(layer._modules[key],
                         name + '.' + key,
                         hook_handler,
                         show=show))
    return handles


def get_hook(layer, name, show=False):
    hooks_got = []
    keys = list(layer._modules.keys())
    if keys == []:
        if show:
            print(name, 'hook has been fetched')
        hooks_got.append((name, layer.hook_out))
        return hooks_got
    else:
        for key in keys:
            hooks_got.extend(
                get_hook(layer._modules[key], name + '.' + key, show=show))
    return hooks_got
