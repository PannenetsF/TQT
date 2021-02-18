# TQT
TQT's pytorch implementation.

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

Much often we need to re-train a network, and we can do a quick job with `lambda`. 