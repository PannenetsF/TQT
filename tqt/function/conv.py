"""
Provide quantilized form of torch.nn.modules.conv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .number import qsigned


class Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 weight_bit_width=8,
                 bias_bit_width=16,
                 inter_bit_width=32,
                 retrain=True,
                 quant=False):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode)
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

    def conv_forward(self, input):
        weight = qsigned(self.weight, self.weight_log2_t,
                         self.weight_bit_width)
        if self.bias is not None:
            bias = qsigned(self.bias, self.bias_log2_t, self.bias_bit_width)
        else:
            bias = None
        inter = F.conv2d(input, weight, bias, self.stride, self.padding,
                         self.dilation, self.groups)

        return inter

    def conv_forward_unquant(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv_forward(
            input) if self.quant else self.conv_forward_unquant(input)


if __name__ == '__main__':
    conv = Conv2d(3, 6, 3)
    x = torch.rand(4, 3, 5, 5)
    print(conv(x).shape)
