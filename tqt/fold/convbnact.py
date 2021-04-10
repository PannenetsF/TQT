import torch
import torch.nn as nn
from ..function import qsigned, qunsigned
from .foldmodule import _FoldModule


class Conv2dBNReLU(_FoldModule):
    def __init__(self, conv, bn, relu):
        super().__init__()
        if isinstance(conv, nn.Conv2d) and isinstance(
                bn, nn.BatchNorm2d) and (isinstance(
                    relu, nn.ReLU), isinstance(relu, nn.ReLU6)):
            self.conv = conv
            self.bn = bn
            self.relu = relu
            self.weight_bit_width = self.conv.weight_bit_width
            self.weight_log2_t = self.conv.weight_log2_t
            self.bias_bit_width = self.bn.bias_bit_width
            self.bias_log2_t = self.bn.bias_log2_t
            self.acti_bit_width = self.relu.acti_bit_width
            self.acti_log2_t = self.relu.acti_log2_t
        else:
            raise Exception('The folded function does not meet type check')
        self.bn_freezing = False
        self.quant = False

    def bn_freeze(self, mode=True):
        self.bn_freezing = mode

    def quantilize(self):
        self.quant = True
        self.weight_log2_t.requires_grad = True
        self.bias_log2_t.requires_grad = True

    def floatilize(self):
        self.quant = False
        self.weight_log2_t.requires_grad = False
        self.bias_log2_t.requires_grad = False

    def forward(self, input):
        if self.bn_freezing:
            bn_var = self.bn.running_var.detach().clone().data.reshape(
                -1, 1, 1, 1)
            bn_mean = self.bn.running_mean.detach().clone().data.reshape(
                -1, 1, 1, 1)
            bn_weight = self.bn.weight.detach().clone().data.reshape(
                -1, 1, 1, 1)
            bn_bias = self.bn.bias.detach().clone().data.reshape(-1, 1, 1, 1)
        else:
            bn_var = self.bn.running_var.reshape(-1, 1, 1, 1)
            bn_mean = self.bn.running_mean.reshape(-1, 1, 1, 1)
            bn_weight = self.bn.weight.reshape(-1, 1, 1, 1)
            bn_bias = self.bn.bias.reshape(-1, 1, 1, 1)
        if self.conv.bias is not None:
            conv_bias = self.conv.bias.reshape(-1, 1, 1, 1)
        else:
            conv_bias = 0.

        conv_weight = self.conv.weight * bn_weight / (bn_var +
                                                      self.bn.eps).sqrt()
        conv_bias = bn_weight * (conv_bias - bn_mean) / (
            bn_var + self.bn.eps).sqrt() + bn_bias

        if self.quant:
            conv_weight = qsigned(conv_weight, self.weight_log2_t,
                                  self.weight_bit_width)
            conv_bias = qsigned(conv_bias, self.bias_log2_t,
                                self.bias_bit_width)

        inter = nn.functional.conv2d(input, conv_weight, conv_bias.reshape(-1),
                                     self.conv.stride, self.conv.padding,
                                     self.conv.dilation, self.conv.groups)
        inter = self.relu(inter)

        if self.quant:
            inter = qunsigned(inter, self.acti_log2_t, self.acti_bit_width)

        return inter
