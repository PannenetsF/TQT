import torch
import torch.nn as nn


def threshold_weight_max(module, qmodule, eps=1e-8):
    max_value = torch.tensor([module.weight.abs().flatten().max().data]) + eps
    qmodule.weight_log2_t = torch.nn.Parameter(
        torch.log2(max_value)) if qmodule.retrain else torch.log2(max_value)


def threshold_bias_max(module, qmodule, eps=1e-8):
    max_value = torch.tensor([module.bias.abs().flatten().max().data]) + eps
    qmodule.bias_log2_t = torch.nn.Parameter(
        torch.log2(max_value)) if qmodule.retrain else torch.log2(max_value)


def threshold_activation_max(module, qmodule, eps=1e-8):
    max_value = torch.tensor([module.hook_out.abs().flatten().max().data
                              ]) + eps
    qmodule.acti_log2_t = torch.nn.Parameter(
        torch.log2(max_value)) if qmodule.retrain else torch.log2(max_value)
