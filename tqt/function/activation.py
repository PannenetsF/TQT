"""
Provide quantilized form of torch.nn.modules.activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import qsigned, qunsigned


class ReLU(nn.ReLU):
    def __init__(self, inplace=False, acti_bit_width=8, retrain=True):
        super().__init__(inplace)
        self.acti_bit_width = acti_bit_width
        self.retrain = retrain
        if retrain is True:
            self.acti_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.acti_log2_t = torch.Tensor(1)

    def relu_forward(self, input):
        return qunsigned(F.relu(input), self.acti_log2_t, self.acti_bit_width)

    def forward(self, input):
        return self.relu_forward(input)


class ReLU6(nn.ReLU6):
    def __init__(self, inplace=False, acti_bit_width=8, retrain=True):
        super().__init__(inplace)
        self.acti_bit_width = acti_bit_width
        self.retrain = retrain
        if retrain is True:
            self.acti_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.acti_log2_t = torch.Tensor(1)

    def relu6_forward(self, input):
        return qunsigned(F.relu6(input), self.acti_log2_t, self.acti_bit_width)

    def forward(self, input):
        return self.relu6_forward(input)