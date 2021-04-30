import torch
import torch.nn as nn
from ..function import qsigned, qunsigned
from ..function.extra import SEConv2d, Adder2d, adder2d_function
from .foldmodule import _FoldModule
from .utils import isShiftAdder


class SA2dBNReLU(_FoldModule):
    def __init__(self, shiftadder, bn, relu):
        super().__init__()
        if isShiftAdder(shiftadder) and isinstance(
                bn, nn.BatchNorm2d) and (isinstance(relu, nn.ReLU)
                                         or isinstance(relu, nn.ReLU6)):
            self.shift = shiftadder[0]
            self.adder = shiftadder[1]
            self.bn = bn
            self.relu = nn.ReLU() if isinstance(relu, nn.ReLU) else nn.ReLU6()
            self.bn_weight(bn)
            self.adder_weight(shiftadder[1])
            self.acti_bit_width = relu.acti_bit_width
            self.acti_log2_t = relu.acti_log2_t
        self.bn_freezing = False
        self.quant = False

    def adder_weight(self, adder):
        self.adder_weight_bit_width = adder.weight_bit_width
        self.adder_weight_log2_t = adder.weight.abs().max().detach().data.log2(
        )

    def bn_weight(self, bn):
        bn_var = bn.running_var.detach().clone().data.reshape(-1, 1, 1, 1)
        bn_mean = bn.running_mean.detach().clone().data.reshape(-1, 1, 1, 1)
        bn_weight = bn.weight.detach().clone().data.reshape(-1, 1, 1, 1)
        bn_bias = bn.bias.detach().clone().data.reshape(-1, 1, 1, 1)
        bn_weight = bn_weight / (bn_var + bn.eps).sqrt()
        bn_bias = bn_weight * (-bn_mean) / (bn_var + bn.eps).sqrt() + bn_bias
        self.bn_weight_bit_width = bn.weight_bit_width
        self.bn_weight_log2_t = torch.nn.Parameter(
            bn_weight.abs().max().detach().data.log2())
        self.bn_bias_bit_width = bn.bias_bit_width
        self.bn_bias_log2_t = torch.nn.Parameter(
            bn_bias.abs().max().detach().data.log2())

    def bn_freeze(self, mode=True):
        self.bn_freezing = mode

    def quantilize(self):
        self.quant = True
        self.bn_weight_log2_t.requires_grad = True
        self.adder_weight_log2_t.requires_grad = True
        self.bn_bias_log2_t.requires_grad = True
        self.acti_log2_t.requires_grad = True

    def floatilize(self):
        self.quant = False
        self.bn_weight_log2_t.requires_grad = False
        self.adder_weight_log2_t.requires_grad = False
        self.bn_bias_log2_t.requires_grad = False
        self.acti_log2_t.requires_grad = False

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
        bn_weight = bn_weight / (bn_var + self.bn.eps).sqrt()
        bn_bias = bn_weight * (-bn_mean) / (bn_var +
                                            self.bn.eps).sqrt() + bn_bias
        if self.quant and self.bn_freezing:
            adder_weight = qsigned(self.adder.weight, self.adder_weight_log2_t,
                                   self.adder_weight_bit_width)
            bn_weight = qsigned(bn_weight, self.bn_weight_log2_t,
                                self.bn_weight_bit_width)
            bn_bias = qsigned(bn_bias, self.bn_bias_log2_t,
                              self.bn_bias_bit_width)
            inter = self.shift(input)
            inter = adder2d_function(inter,
                                     adder_weight,
                                     stride=self.adder.stride,
                                     padding=self.adder.padding)
            inter = bn_weight * inter + bn_bias
        elif self.quant and self.bn_freezing == False:
            adder_weight = qsigned(self.adder.weight, self.adder_weight_log2_t,
                                   self.adder_weight_bit_width)
            inter = self.shift(input)
            inter = adder2d_function(inter,
                                     adder_weight,
                                     stride=self.adder.stride,
                                     padding=self.adder.padding)
            inter = self.bn(inter)
        else:
            inter = self.shift(input)
            inter = adder2d_function(inter, self.adder.weight,
                                     self.adder.stride, self.adder.padding)
            inter = self.bn(inter)

        inter = self.relu(inter)

        if self.quant:
            inter = qunsigned(inter, self.acti_log2_t, self.acti_bit_width)
        return inter
