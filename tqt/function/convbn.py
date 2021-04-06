"""
Provide quantilized form of torch.nn.modules.conv and bn2d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .number import qsigned


class Conv2dBN(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 padding_mode='zeros',
                 weight_bit_width=8,
                 bias_bit_width=16,
                 inter_bit_width=32,
                 retrain=True,
                 quant=False,
                 fold=False):
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
        self.fold = fold
        self.bn = nn.BatchNorm2d(out_channels,
                                 eps=eps,
                                 momentum=momentum,
                                 affine=affine,
                                 track_running_stats=track_running_stats)
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
        self.weight_log2_t.requires_grad = False
        if self.bias is not None:
            self.bias_log2_t.requires_grad = False

    def floatilize(self):
        self.quant = False
        self.weight_log2_t.requires_grad = True
        if self.bias is not None:
            self.bias_log2_t.requires_grad = True

    def make_fold(self, mode=True):
        self.fold = mode

    def quant_answer(self):
        self.weight.data = qsigned(
            self.weight, self.weight_log2_t,
            self.weight_bit_width)**(self.weight_bit_width - 1 -
                                     torch.ceil(self.weight_log2_t)).int()
        if self.bias is not None:
            self.bias.data = qsigned(
                self.bias, self.bias_log2_t,
                self.bias_bit_width)**(self.bias_bit_width - 1 -
                                       torch.ceil(self.bias_log2_t)).int()
        self.bn.weight = qsigned(
            self.bn.weight, self.weight_log2_t,
            self.weight_bit_width)**(self.weight_bit_width - 1 -
                                     torch.ceil(self.weight_log2_t)).int()
        self.bn.bias = qsigned(
            self.bn.bias, self.bias_log2_t,
            self.bias_bit_width)**(self.bias_bit_width - 1 -
                                   torch.ceil(self.bias_log2_t)).int()

    def conv_bn_forward(self, input):
        if self.fold:
            bn_running_var = self.bn.running_var.unsqueeze(1).unsqueeze(
                2).unsqueeze(3)
            bn_weight = self.bn.weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            weight_bn = bn_weight * self.weight / (bn_running_var +
                                                   self.bn.eps).sqrt()
            weight = qsigned(weight_bn, self.weight_log2_t,
                             self.weight_bit_width)
            if self.bias is None:
                bias_bn = self.bn.bias - self.bn.weight * self.bn.running_mean / (
                    self.bn.running_var + self.bn.eps).sqrt()
                bias = qsigned(bias_bn, self.bias_log2_t, self.bias_bit_width)
            else:
                bias_bn = self.bn.bias + self.bn.weight * (
                    self.bias - self.bn.running_mean) / (self.bn.running_var +
                                                         self.bn.eps).sqrt()
                bias = qsigned(bias_bn, self.bias_log2_t, self.bias_bit_width)
        else:
            weight = qsigned(self.weight, self.weight_log2_t,
                             self.weight_bit_width)
            if self.bias is not None:
                bias = qsigned(self.bias, self.bias_log2_t,
                               self.bias_bit_width)
            else:
                bias = 0.
        input_log2_t = input.abs().max().log2()
        inter = qsigned(
            F.conv2d(input, weight, None, self.stride, self.padding,
                     self.dilation, self.groups),
            self.weight_log2_t + input_log2_t + math.log2(self.weight.numel()),
            self.inter_bit_width)
        inter += bias
        return inter

    def conv_bn_forward_unquant(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv_bn_forward(
            input) if self.quant else self.conv_bn_forward_unquant(input)


if __name__ == '__main__':
    conv = Conv2d(3, 6, 3)
    x = torch.rand(4, 3, 5, 5)
    print(conv(x).shape)
