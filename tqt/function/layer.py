import torch
import torch.nn as nn
from .number import qsigned, qunsigned


class SignedLayer(nn.Module):
    def __init__(self, output_bit_width=16, retrain=True, quant=False):
        self.output_bit_width = output_bit_width
        self.retrain = retrain
        self.quant = quant
        if retrain is True:
            self.output_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.output_log2_t = torch.Tensor(1)

    def static(self):
        self.retrain = False
        if isinstance(self.output_log2_t, nn.Parameter):
            self.output_log2_t.requires_grad_(False)

    def quantilize(self):
        self.quant = True

    def floatilize(self):
        self.quant = False

    def forward(self, input):
        return qsigned(input, self.output_log2_t, self.output_bit_width)


class UnsignedLayer(nn.Module):
    def __init__(self, output_bit_width=16, retrain=True, quant=False):
        self.output_bit_width = output_bit_width
        self.retrain = retrain
        self.quant = quant
        if retrain is True:
            self.output_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.output_log2_t = torch.Tensor(1)

    def static(self):
        self.retrain = False
        if isinstance(self.output_log2_t, nn.Parameter):
            self.output_log2_t.requires_grad_(False)

    def quantilize(self):
        self.quant = True

    def floatilize(self):
        self.quant = False

    def forward(self, input):
        return qunsigned(input, self.output_log2_t, self.output_bit_width)
