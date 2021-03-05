import torch
import torch.nn as nn


def threshold_weight_3sd(module, qmodule, eps=1e-8):
    mean_value = torch.tensor([module.weight.mean().abs().data])
    std_value = module.weight.std().data
    qmodule.weight_log2_t = torch.nn.Parameter(
        torch.log2(mean_value +
                   3 * std_value)) if qmodule.retrain else torch.log2(
                       mean_value + 3 * std_value + eps)


def threshold_bias_3sd(module, qmodule, eps=1e-8):
    mean_value = torch.tensor([module.bias.mean().abs().data])
    std_value = module.bias.std().data
    qmodule.bias_log2_t = torch.nn.Parameter(
        torch.log2(mean_value +
                   3 * std_value)) if qmodule.retrain else torch.log2(
                       mean_value + 3 * std_value + eps)


def threshold_activation_3sd(module, qmodule, eps=1e-8):
    mean_value = torch.tensor([module.hook_out.mean().abs().data])
    std_value = module.hook_out.std().data
    qmodule.acti_log2_t = torch.nn.Parameter(
        torch.log2(mean_value +
                   3 * std_value)) if qmodule.retrain else torch.log2(
                       mean_value + 3 * std_value + eps)
