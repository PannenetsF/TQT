"""
Provide quantilized form of torch.nn.modules.linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import qsigned
from .layer import SignedLayer


class _Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit_width=8,
                 bias_bit_width=16,
                 inter_bit_width=16,
                 retrain=True,
                 quant=False):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width
        self.inter_bit_width = inter_bit_width
        self.dirty_hook = None
        self.retrain = retrain
        self.quant = quant
        if retrain is True:
            self.weight_log2_t = nn.Parameter(torch.Tensor(1))
            self.inter_log2_t = nn.Parameter(torch.Tensor(1))
            if self.bias is not None:
                self.bias_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.weight_log2_t = torch.Tensor(1)
            self.inter_log2_t = torch.Tensor(1)
            if self.bias is not None:
                self.bias_log2_t = torch.Tensor(1)

    def static(self):
        self.retrain = False
        if isinstance(self.bias_log2_t, nn.Parameter):
            self.bias_log2_t.requires_grad_(False)
        if isinstance(self.weight_log2_t, nn.Parameter):
            self.weight_log2_t.requires_grad_(False)

    def quantilize(self):
        self.quant = True

    def floatilize(self):
        self.quant = False

    def linear_forward(self, input):
        bias = None
        weight = qsigned(self.weight, self.weight_log2_t,
                         self.weight_bit_width)
        inter = qsigned(F.linear(input, weight, bias), self.inter_log2_t,
                        self.inter_bit_width)
        if self.dirty_hook is not None:
            self.dirty_hook_out = inter
        if self.bias is not None:
            inter += qsigned(self.bias, self.bias_log2_t,
                             self.bias_bit_width).reshape(1, -1)
        return inter

    def linear_forward_unquant(self, input):
        return F.linear(input, self.weight, self.bias)

    def forward(self, input):
        return self.linear_forward(
            input) if self.quant else self.linear_forward_unquant(input)


class Linear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit_width=8,
                 bias_bit_width=16,
                 inter_bit_width=16,
                 output_bit_width=8,
                 retrain=True,
                 quant=False):
        super().__init__()
        self.inter_bit_width = inter_bit_width
        self.output_bit_width = output_bit_width
        self.linear = _Linear(in_features,
                              out_features,
                              bias=bias,
                              weight_bit_width=weight_bit_width,
                              bias_bit_width=bias_bit_width,
                              inter_bit_width=inter_bit_width,
                              retrain=retrain,
                              quant=quant)
        self.quantlayer = SignedLayer(output_bit_width=output_bit_width,
                                      retrain=retrain,
                                      quant=quant)

    def forward(self, input):
        input = self.linear(input)
        input = self.quantlayer(input)
        return input


if __name__ == '__main__':
    lin = Linear(3, 5)
    x = torch.rand(3, 3)
    print(lin(x).shape)
