"""
Provide quantilized form of torch.nn.modules.activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import qsigned, qunsigned


class ReLU(nn.ReLU):
    def __init__(self,
                 inplace=False,
                 acti_bit_width=8,
                 retrain=True,
                 quant=False):
        super().__init__(inplace)
        self.acti_bit_width = acti_bit_width
        self.retrain = retrain
        self.quant = quant
        if retrain is True:
            self.acti_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.acti_log2_t = torch.Tensor(1)

    def relu_forward(self, input):
        return qunsigned(F.relu(input), self.acti_log2_t, self.acti_bit_width)

    def relu_forward_unquant(self, input):
        return F.relu(input)

    def quantilize(self):
        self.quant = True
        self.acti_log2_t.requires_grad = False

    def floatilize(self):
        self.quant = False
        self.acti_log2_t.requires_grad = True

    def forward(self, input):
        return self.relu_forward(
            input) if self.quant else self.relu_forward_unquant(input)


class ReLU6(nn.ReLU6):
    def __init__(self,
                 inplace=False,
                 acti_bit_width=8,
                 retrain=True,
                 quant=False):
        super().__init__(inplace)
        self.acti_bit_width = acti_bit_width
        self.retrain = retrain
        self.quant = quant
        if retrain is True:
            self.acti_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.acti_log2_t = torch.Tensor(1)

    def relu6_forward(self, input):
        return qunsigned(F.relu6(input), self.acti_log2_t, self.acti_bit_width)

    def relu6_forward_unquant(self, input):
        return F.relu6(input)

    def quantilize(self):
        self.quant = True

    def floatilize(self):
        self.quant = False

    def forward(self, input):
        return self.relu6_forward(
            input) if self.quant else self.relu6_forward_unquant(input)
