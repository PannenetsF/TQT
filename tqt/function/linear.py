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
        self.weight_log2_t.requires_grad = True
        if self.bias is not None:
            self.bias_log2_t.requires_grad = True

    def floatilize(self):
        self.quant = False
        self.weight_log2_t.requires_grad = False
        if self.bias is not None:
            self.bias_log2_t.requires_grad = False

    def quant_answer(self):
        self.weight.data = qsigned(
            self.weight, self.weight_log2_t,
            self.weight_bit_width)**(self.weight_bit_width - 1 -
                                     torch.ceil(self.weight_log2_t)).int()
        self.bias.data = qsigned(
            self.bias, self.bias_log2_t,
            self.bias_bit_width)**(self.bias_bit_width - 1 -
                                   torch.ceil(self.bias_log2_t)).int()

    def linear_forward(self, input):
        weight = qsigned(self.weight, self.weight_log2_t,
                         self.weight_bit_width)
        if self.bias is not None:
            bias = qsigned(self.bias, self.bias_log2_t, self.bias_bit_width)
        else:
            bias = None
        inter = F.linear(input, weight, bias)
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
