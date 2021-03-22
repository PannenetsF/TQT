"""
Provide quantilized form of Adder2d, https://arxiv.org/pdf/1912.13200.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from . import extra as ex
from .number import qsigned
from .layer import SignedLayer


class _Adder2d(ex.Adder2d):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 weight_bit_width=8,
                 bias_bit_width=16,
                 inter_bit_width=8,
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
        self.inter_bit_width = inter_bit_width
        self.retrain = retrain
        self.quant = quant
        self.dirty_hook = None
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
        pass

    def quantilize(self):
        self.quant = True

    def floatilize(self):
        self.quant = False

    def adder_forward(self, input):
        weight = qsigned(self.weight, self.weight_log2_t,
                         self.weight_bit_width)
        inter = qsigned(
            ex.adder2d_function(input,
                                weight,
                                bias=None,
                                stride=self.stride,
                                padding=self.padding), self.inter_log2_t,
            self.inter_bit_width)
        if self.bias is not None:
            inter += qsigned(self.bias, self.bias_log2_t,
                             self.bias_bit_width).reshape(1, -1, 1, 1)
        return inter

    def adder_forward_unquant(self, input):
        inter = ex.adder2d_function(input,
                                    self.weight,
                                    bias=None,
                                    stride=self.stride,
                                    padding=self.padding)
        if self.dirty_hook is not None:
            self.dirty_hook_out = inter
        if self.bias is not None:
            inter += self.bias.reshape(1, -1, 1, 1)
        return inter

    def forward(self, input):
        return self.adder_forward(
            input) if self.quant else self.adder_forward_unquant(input)


class Adder2d(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 weight_bit_width=8,
                 bias_bit_width=16,
                 inter_bit_width=16,
                 output_bit_width=8,
                 retrain=True,
                 quant=False):
        super().__init__()
        self.adder = _Adder2d(input_channel,
                              output_channel,
                              kernel_size,
                              stride=stride,
                              padding=padding,
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
        input = self.adder(input)
        input = self.quantlayer(input)
        return input


if __name__ == '__main__':
    add = Adder2d(3, 4, 3, bias=True)
    x = torch.rand(10, 3, 10, 10)
    print(add(x).shape)
