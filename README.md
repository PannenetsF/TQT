# TQT
TQT's pytorch implementation.

## TQT.function 

`function` is a re-impletement of `torch.nn.modules`. Besides all the args used in the original function, a quantilized function get 2 kind of optional arguments: `bit_width` and `retrain`. 

If the `retrain` is `True`, the Module will be in Retrain Mode, with the `log2_t` trainable. Else, in Static Mode, the `log2_t` are determined by initialization and not trainable.