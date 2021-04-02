"""
Provide quantilized form of torch.nn.modules.linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .number import qsigned


class Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit_width=8,
                 bias_bit_width=16,
                 inter_bit_width=32,
                 retrain=True,
                 quant=False):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width
        self.inter_bit_width = inter_bit_width
        self.retrain = retrain
        self.quant = quant
        if retrain is True:
            self.weight_log2_t = nn.Parameter(torch.Tensor(1))
            if self.bias is not None:
                self.bias_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.weight_log2_t = torch.Tensor(1)
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
        input_log2_t = input.abs().max().log2()
        weight = qsigned(self.weight, self.weight_log2_t,
                         self.weight_bit_width)
        inter = qsigned(
            F.linear(input, weight, None),
            self.weight_log2_t + input_log2_t + math.log2(self.weight.numel()),
        if self.bias is not None:
            inter += qsigned(self.bias, self.bias_log2_t,
                             self.bias_bit_width).unsqueeze(0)
        return inter

    def linear_forward_unquant(self, input):
        return F.linear(input, self.weight, self.bias)

    def forward(self, input):
        return self.linear_forward(
            input) if self.quant else self.linear_forward_unquant(input)


if __name__ == '__main__':
    lin = Linear(3, 5)
    x = torch.rand(3, 3)
    print(lin(x).shape)
