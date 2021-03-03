num_list = [str(i) for i in range(0, 10)]


def _getattr(obj, key):
    stage = key.rsplit('.')
    for i in stage:
        if i[0] in num_list:
            obj = obj[int(i)]
        else:
            obj = getattr(obj, i)
    return obj


def make_net_quant_or_not(net,
                          quant=True,
                          module_name_preop=lambda x, net=None: x):
    okeys = list(net.state_dict().keys())
    okeys = [module_name_preop(x, net) for x in okeys]
    mkeys = [x[:x.rfind('.')] for x in okeys]
    for key in mkeys:
        net_proc = _getattr(net, key)
        if hasattr(net_proc, 'quant'):
            getattr(net_proc, 'quantilize')() if quant else getattr(
                net_proc, 'floatilize')
            print(net_proc)
