# TQT
TQT's pytorch implementation.

- [TQT](#tqt)
  - [TQT's modules](#tqts-modules)
    - [TQT.function](#tqtfunction)
    - [TQT.threshold](#tqtthreshold)
  - [Build a network with TQT](#build-a-network-with-tqt)
  - [Rebuild a network with TQT](#rebuild-a-network-with-tqt)
  - [Initialize a network's threshold](#initialize-a-networks-threshold)
  - [Train Something with Pre-Trained Model](#train-something-with-pre-trained-model)
  - [Do analyse over the activations and weights](#do-analyse-over-the-activations-and-weights)

## TQT's modules

### TQT.function 

`function` is a re-impletement of `torch.nn.modules`. Besides all the args used in the original function, a quantilized function get 2 kind of optional arguments: `bit_width` and `retrain`. 

`bit_width` has 2 type: weight/bias or activation. 

If the `retrain` is `True`, the Module will be in Retrain Mode, with the `log2_t` trainable. Else, in Static Mode, the `log2_t` are determined by initialization and not trainable.

### TQT.threshold

Provide 3 ways to initialize the threshold: `init_max`, `init_kl_j`, `init_3sd`. 

To initialize the weight and threshold correctly, please follow the method to build a network with TQT.

## Build a network with TQT

To get output of each tqt module, the network should be flat, that is, no `nn.Sequential`, no nested `nn.ModuleList`. 

You'd better use `nn.ModuleList` and append every operation after it. If there're some operations that are `nn.ModuleList` of some operation, you can use `.extend` to keep the network flat. 

## Rebuild a network with TQT 

Much often we need to re-train a network, and we can do a quick job with `lambda`. As you can see in the file `lenet.py`, with the change of the wrapper, a net could be simply converted into a quantilized one. 

## Initialize a network's threshold 

Just 3 steps! 

1. Add hook for output storage.
2. Adjust the threshold via `tqt.threshold` 
3. Remove hook.

## Train Something with Pre-Trained Model

Supposed that you have a pretrained model, and it's hard to change all keys in its state dictionary. More often, it may contain lots of `nn.Module` but not specially `nn.ModuleList`. A dirty but useful way is simply change the `import torch.nn as nn` to `import tqt.function as nn`. You can get a quant-style network with all previous keys unchanged! 

All you need to do is add a list `self.proc` to the network module.

Through `tqt.threshold.add_hook_general`, we can add hook for any network if you add a list containing all operations used in forward.

Let's get some example: 

```py
# noquant.py
import torch.nn as nn 

class myNet(nn.Module):
    def __init__(self, args):
        # assume: all op used in forward are declared explicitly.
        self.op1 = ... 
        self.op2 = ...
        if args:
            self.op_args = ...
        ...
    def forward(self, x):
        ...
```

and

```py
# quant.py
import tqt.function as nn 

class myNet(nn.Module):
    def __init__(self):
        # assume: all op used in forward are declared explicitly.
        self.proc = ['op1', 'op2']
        self.op1 = ... 
        self.op2 = ...
        if args:
            self.op_args = ...
            self.proc.append('op_args')
        ...
    def forward(self, x):
        ...
```

We can load and retrain by:

```py
# main.py 
import tqt
from unquant import myNet as oNet
from quant import myNet as qNet

handler = tqt.threshold.hook_handler

train(oNet) ... 
funct_list = [oNet.xx, oNet.yy, ...]
qfunct_list = [qNet.xx, qNet.yy, ...]
for funct in funct_list:
    handles = tqt.threshold.add_hook_general(funct, handler)
qNet.load_state_dict(oNet.state_dict(), strict=False)
for (netproc, qnetproc) in zip(funct_list, qfunct_list):
    tqt.threshold.init.init_network(netproc, qnetproc, show=True)
retrain(qNet)
```

## Do analyse over the activations and weights

Always, we need to do analysis over activations and weights to choose a proper way to quantilize the network. We implement some function do these. It's recommend do this with tensorboard.

`tqt.threshold.get_hook` will get all hook output got from the forward with their module name as a tuple. 

```py
net = QNet()
tqt.utils.make_net_quant_or_not(net, quant=True)
tqt.threshold.add_hook_general(net, tqt.threshold.hook_handler)
net.cuda()
for i, (images, labels) in enumerate(data_test_loader):
    net(images.cuda())
    break
out = get_hook(net, '', show=True
for i in out:
    print(i[0], i[1].shape)
writer.add_histogram(i[0], i[1].cpu().data.flatten().detach().numpy())
```

Similarly, the weights could be get from `net.named_parameters()`.