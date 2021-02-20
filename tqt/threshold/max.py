import torch
import torch.nn as nn


def threshold_weight_max(module, qmodule):
    max_value = torch.tensor([module.weight.abs().flatten().max().data])
    qmodule.weight_log2_t = torch.nn.Parameter(
        torch.log2(max_value)) if qmodule.retrain else torch.log2(max_value)


def threshold_bias_max(module, qmodule):
    max_value = torch.tensor([module.bias.abs().flatten().max().data])
    qmodule.bias_log2_t = torch.nn.Parameter(
        torch.log2(max_value)) if qmodule.retrain else torch.log2(max_value)


def threshold_activation_max(module, qmodule):
    max_value = torch.tensor([module.hook_out.abs().flatten().max().data])
    qmodule.acti_log2_t = torch.nn.Parameter(
        torch.log2(max_value)) if qmodule.retrain else torch.log2(max_value)
