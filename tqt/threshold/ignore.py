import torch
import torch.nn as nn


def threshold_weight_ig(module, qmodule, eps=1e-8):
    ig_value = torch.tensor([0.])
    qmodule.weight_log2_t = torch.nn.Parameter(
        ig_value) if qmodule.retrain else ig_value


def threshold_bias_ig(module, qmodule, eps=1e-8):
    ig_value = torch.tensor([qmodule.bias_bit_width / 2.]) + eps
    qmodule.bias_log2_t = torch.nn.Parameter(
        ig_value) if qmodule.retrain else ig_value


def threshold_activation_ig(module, qmodule, eps=1e-8):
    ig_value = torch.tensor([qmodule.acti_bit_width - 1.]) + eps
    qmodule.acti_log2_t = torch.nn.Parameter(
        ig_value) if qmodule.retrain else ig_value
