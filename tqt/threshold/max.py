import torch
import torch.nn as nn


def threshold_weight_max(module):
    max_value = module.weight.abs().flatten().max().data
    module.weight_log2_t = torch.log2(max_value)


def threshold_bias_max(module):
    max_value = module.bias.abs().flatten().max().data
    module.bias_log2_t = torch.log2(max_value)


def threshold_activation_max(module):
    max_value = module.hook_out.abs().flatten().max().data
    module.acti_log2_t = torch.log2(max_value)
