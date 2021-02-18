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
