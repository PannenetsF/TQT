# TQT
TQT's pytorch implementation.

- [TQT](#tqt)
  - [Notice](#notice)
  - [TQT's modules](#tqts-modules)
    - [TQT.function](#tqtfunction)
    - [TQT.threshold](#tqtthreshold)
  - [Build a network with TQT](#build-a-network-with-tqt)
  - [Rebuild a network with TQT](#rebuild-a-network-with-tqt)
  - [Initialize a network's threshold](#initialize-a-networks-threshold)
  - [Train Something with Pre-Trained Model](#train-something-with-pre-trained-model)
  - [Turn a network to quantized or not](#turn-a-network-to-quantized-or-not)
  - [Exclude some types of module](#exclude-some-types-of-module)
  - [Do analyse over the activations and weights](#do-analyse-over-the-activations-and-weights)
- [Contributing](#contributing)
- [Acknowledgment](#acknowledgment)

## Notice

Now availabel at  [https://pypi.org/project/tqt/0.1.2/](https://pypi.org/project/tqt/0.1.0/)!


Networks quantized via this package could be find at [https://github.com/PannenetsF/QuantizationPool](https://github.com/PannenetsF/QuantizationPool).


## TQT's modules

### TQT.function 

`function` is a re-impletement of `torch.nn.modules`. Besides all the args used in the original function, a quantized function get 2 kind of optional arguments: `bit_width` and `retrain`. 

`bit_width` has 2 type: weight/bias or activation. 

If the `retrain` is `True`, the Module will be in Retrain Mode, with the `log2_t` trainable. Else, in Static Mode, the `log2_t` are determined by initialization and not trainable.

### TQT.threshold

Provide 3 ways to initialize the threshold: `init_max`, `init_kl_j`, `init_3sd`. 

To initialize the weight and threshold correctly, please follow the method to build a network with TQT.
`xxxxx.xxx`. 

As we know, the input(k1-b-m1-p) multiplied by weights(k2-b-m2-p) will be like (k1+k2)-b-(m1+m2)-p. And then we need to accumlate all these inter-output. Assuming there are n inter-output, the bitwidth will need at least `ceil(log2(n))` more bits to make sure the data should not overflow. For a typical network, a 3x3x64 kernel will call for 10 more bits, then the inter-output is (k1+k2+10)-b-(m1+m2)-p. Considering the 2-pow, it will be better to use 32 bit. 

So in the code, we will have:

```py
inter = qsigned(inter, self.weight_log2_t + input_log2_t, self.inter_bit_width) 
```

But in need of keep the network unchanged, we cannot treat `input_log2_t` as a argument. Then

```py 
input_log2_t = math.ceil(math.log2(math.ceil(input.max())))
```

## Build a network with TQT

To get output of each tqt module, the network should be flat, that is, no `nn.Sequential`, no nested `nn.ModuleList`. 

You'd better use `nn.ModuleList` and append every operation after it. If there're some operations that are `nn.ModuleList` of some operation, you can use `.extend` to keep the network flat. 

## Rebuild a network with TQT 

Much often we need to re-train a network, and we can do a quick job with `lambda`. As you can see in the file `lenet.py`, with the change of the wrapper, a net could be simply converted into a quantized one. 

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
tqt.threshold.add_hook(oNet, 'oNet', handler)
qNet.load_state_dict(oNet.state_dict(), strict=False)
for (netproc, qnetproc) in zip(funct_list, qfunct_list):
    tqt.threshold.init.init_network(netproc, qnetproc, show=True)
retrain(qNet)
```

## Turn a network to quantized or not

With a network built by [method metioned](#train-something-with-pre-trained-model), we may need use a quant/or-not version. So we implement `tqt.utils.make_net_quant_or_not` to change its mode easily.

## Exclude some types of module

Normally we wil disable the quantization of batchnorm modules, you can simply exclude the bn in `tqt.utils.make_net_quant_or_not` like:

```py
tqt.utils.make_net_quant_or_not(net,
                                'net',
                                quant=True,
                                exclude=[torch.nn.BatchNorm2d],
                                show=True)
```

## Do analyse over the activations and weights

Always, we need to do analysis over activations and weights to choose a proper way to quantize the network. We implement some function do these. It's recommend do this with tensorboard.

`tqt.threshold.get_hook` will get all hook output got from the forward with their module name as a tuple. 

```py
net = QNet()
tqt.utils.make_net_quant_or_not(net, quant=True)
tqt.threshold.add_hook(net, 'net', tqt.threshold.hook_handler)
net.cuda()
for i, (images, labels) in enumerate(data_test_loader):
    net(images.cuda())
    break
out = get_hook(net, 'net', show=True)
for i in out:
    print(i[0], i[1].shape)
writer.add_histogram(i[0], i[1].cpu().data.flatten().detach().numpy())
```

Similarly, the weights could be get from `net.named_parameters()`.


# Contributing 

It will be great of you to make this project better! There is some ways to contribute!

1. To start with, issues and feature request could let maintainers know what's wrong or anything essential to be added. 
2. If you use the package in you work/repo, just cite the repo and add a dependency note! 
3. You can add some function in `torch.nn` like `HardTanh` and feel free to open a pull request! The code style is simple as [here](style.md).

# Acknowledgment 

The initial version of tqt-torch is developed by [Jinyu Bai](https://github.com/buaabai). 

The beta version was tested by [Jinghan Xu](https://github.com/Xu-Jinghan), based on whose feedback a lot of bugs were fixed.

The original papar could be find at [Arxiv, Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks](https://arxiv.org/abs/1903.08066).