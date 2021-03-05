"""
Provide quantilized form of Adder2d, https://arxiv.org/pdf/1912.13200.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from . import extra as ex
from .number import qsigned


class Adder2d(ex.Adder2d):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 weight_bit_width=8,
                 bias_bit_width=16,
                 retrain=True,
                 quant=False):
        super().__init__(input_channel,
                         output_channel,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=bias)
        self.weight_bit_width = weight_bit_width
        self.bias_bit_width = bias_bit_width
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
        pass

    def quantilize(self):
        self.quant = True

    def floatilize(self):
        self.quant = False

    def adder_forward(self, input):
        if self.bias is None:
            bias = None
        else:
            bias = qsigned(self.bias, self.bias_log2_t, self.bias_bit_width)
        weight = qsigned(self.weight, self.weight_log2_t,
                         self.weight_bit_width)
        return ex.adder2d_function(input,
                                   weight,
                                   bias,
                                   stride=self.stride,
                                   padding=self.padding)

    def adder_forward_unquant(self, input):
        return ex.adder2d_function(input,
                                   self.weight,
                                   self.bias,
                                   stride=self.stride,
                                   padding=self.padding)

    def forward(self, input):
        return self.adder_forward(
            input) if self.quant else self.adder_forward_unquant(input)


if __name__ == '__main__':
    add = Adder2d(3, 4, 3, bias=True)
    x = torch.rand(10, 3, 10, 10)
    print(add(x).shape)
