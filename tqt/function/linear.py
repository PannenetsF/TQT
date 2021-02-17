"""
Provide quantilized form of torch.nn.modules.linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import qsigned


class Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit_width=8,
                 bias_bit_width=16,
                 retrain=True):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width
        if retrain is True:
            self.weight_log2_t = nn.Parameter(torch.Tensor(1))
            self.bias_log2_t = nn.Parameter(torch.Tensor(1))
            self.init_param(retrain)
        else:
            self.weight_log2_t = torch.Tensor(1)
            self.bias_log2_t = torch.Tensor(1)
            self.init_param(retrain)
        pass

    def init_param(self, retrain):
        pass

    def linear_forward(self, input):
        if self.bias is None:
            bias = None
        else:
            bias = qsigned(self.bias, self.bias_log2_t, self.bias_bit_width)
        weight = qsigned(self.weight, self.weight_log2_t,
                         self.weight_bit_width)
        return F.linear(input, weight, bias)

    def forward(self, input):
        return self.linear_forward(input)


if __name__ == '__main__':
    lin = Linear(3, 5)
    x = torch.rand(3, 3)
    print(lin(x).shape)