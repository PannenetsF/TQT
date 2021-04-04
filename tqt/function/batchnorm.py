"""
Provide quantilized form of torch.nn.modules.batchnorm 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import qsigned


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 weight_bit_width=8,
                 bias_bit_width=16,
                 retrain=True,
                 quant=False):
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats)

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

    def static(self):
        self.retrain = False
        if isinstance(self.bias_log2_t, nn.Parameter):
            self.bias_log2_t.requires_grad_(False)
        if isinstance(self.weight_log2_t, nn.Parameter):
            self.weight_log2_t.requires_grad_(False)

    def quantilize(self):
        self.quant = True
        self.bias_log2_t.requires_grad = True
        self.weight_log2_t.requires_grad = True

    def floatilize(self):
        self.quant = False
        self.bias_log2_t.requires_grad = False
        self.weight_log2_t.requires_grad = False

    def fold(self, pre):
        self.weight_log2_t.data = pre.weight_log2_t.data
        if hasattr(pre, 'bias_log2_t'):
            self.bias_log2_t.data = pre.bias_log2_t.data

    def quant_answer(self):
        self.weight.data = qsigned(
            self.weight, self.weight_log2_t,
            self.weight_bit_width)**(self.weight_bit_width - 1 -
                                     torch.ceil(self.weight_log2_t)).int()
        self.bias.data = qsigned(
            self.bias, self.bias_log2_t,
            self.bias_bit_width)**(self.bias_bit_width - 1 -
                                   torch.ceil(self.bias_log2_t)).int()

    def bn_forward(self, input):
        if self.affine is True:
            weight = qsigned(self.weight, self.weight_log2_t,
                             self.weight_bit_width)
            bias = qsigned(self.bias, self.bias_log2_t, self.bias_bit_width)
            output = F.batch_norm(input,
                                  running_mean=self.running_mean,
                                  running_var=self.running_var,
                                  weight=weight,
                                  bias=bias)
        else:
            output = F.batch_norm(input,
                                  running_mean=self.running_mean,
                                  running_var=self.running_var)
        return output

    def bn_forward_unquant(self, input):
        if self.affine is True:
            output = F.batch_norm(input,
                                  running_mean=self.running_mean,
                                  running_var=self.running_var,
                                  weight=self.weight,
                                  bias=self.bias)
        else:
            output = F.batch_norm(input,
                                  running_mean=self.running_mean,
                                  running_var=self.running_var)
        return output

    def forward(self, input):
        return self.bn_forward(
            input) if self.quant else self.bn_forward_unquant(input)
