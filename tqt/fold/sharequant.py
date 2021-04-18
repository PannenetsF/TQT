import torch
import torch.nn as nn
from ..function import qsigned


class ShareQuant(nn.Module):
    def __init__(self, acti_bit_width=8, retrain=True, quant=False):
        super().__init__()
        self.acti_bit_width = acti_bit_width
        self.acti_log2_t = nn.Parameter(torch.Tensor(1))
        self.retrain = retrain
        self.quant = quant

    def share(self, shared_module):
        self.acti_log2_t.data = max(shared_module.acti_log2_t.data,
                                    self.acti_log2_t.data)

    def static(self):
        self.retrain = False
        self.acti_log2_t.requires_grad_(False)

    def quantilize(self):
        self.quant = True
        self.acti_log2_t.requires_grad_(True)

    def floatilize(self):
        self.quant = False
        self.acti_log2_t.requires_grad_(False)

    def share_forward(self, input):
        return qsigned(input, self.acti_log2_t, self.acti_bit_width)

    def share_forward_unquant(self, input):
        return input

    def forward(self, input):
        return self.share_forward(
            input) if self.quant else self.share_forward_unquant(input)
