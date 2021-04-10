import torch

num_list = [str(i) for i in range(0, 10)]


def _getattr(obj, key):
    stage = key.rsplit('.')
    for i in stage:
        if i[0] in num_list:
            obj = obj[int(i)]
        else:
            obj = getattr(obj, i)
    return obj


def _isinstance(obj, cls_list):
    flag_list = [isinstance(obj, cls) for cls in cls_list]
    flag = False
    for x in flag_list:
        flag = x or flag
    return flag


def make_net_quant_or_not(net_proc, name, quant=True, exclude=[], show=False):
    keys = list(net_proc._modules.keys())
    if hasattr(net_proc, 'quant'):
        if not _isinstance(net_proc, exclude):
            getattr(net_proc, 'quantilize')() if quant else getattr(
                net_proc, 'floatilize')()
            if show:
                print(name, ' is quanted')
        else:
            getattr(net_proc, 'floatilize')()
            if show:
                print(name, 'is excluded')
    for key in keys:
        make_net_quant_or_not(net_proc._modules[key],
                              name + f'.{key}',
                              quant=quant,
                              exclude=exclude,
                              show=show)


def make_bn_fold_with_previous(net_proc, name, pre_proc, pre_name, show=False):
    keys = list(net_proc._modules.keys())
    _pre_proc = None
    _pre_name = ''
    if isinstance(net_proc, torch.nn.BatchNorm2d):
        if pre_proc is not None:
            net_proc.fold(pre_proc)
            if show:
                print(f'{name} is folded with {pre_name}')
    return net_proc, name
    for key in keys:
        _pre_proc, _pre_name = make_bn_fold_with_previous(
            net_proc._modules[key],
            name + f'.{key}',
            _pre_proc,
            _pre_name,
            show=show)
    return _pre_proc, _pre_name
