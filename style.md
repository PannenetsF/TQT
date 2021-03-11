# tqt.function code style

All you need to follow is to change the upper case words and add you module! Then update the `./tqt.function/__init__.py` and `./tqt/wrapper.py`. 

```py
# filename: FILENAME.py
"""
Provide quantilized form of torch.nn.modules.FILENAME 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .number import qsigned, qunsigned


class FUNCTION(nn.FUNCTION):
    def __init__(self,
                 *ORIGINAL_ARGS,
                 acti_bit_width=8,
                 retrain=True,
                 quant=False):
        super().__init__(*ORIGINAL_ARGS)
        self.acti_bit_width = acti_bit_width
        self.retrain = retrain
        self.quant = quant
        # or if there is weight or bias 
        # if retrain is True:
        #     self.weight_log2_t = nn.Parameter(torch.Tensor(1))
        #     if self.bias is not None:
        #         self.bias_log2_t = nn.Parameter(torch.Tensor(1))
        # else:
        #     self.weight_log2_t = torch.Tensor(1)
        #     if self.bias is not None:
        #         self.bias_log2_t = torch.Tensor(1) 
        if retrain is True:
            self.acti_log2_t = nn.Parameter(torch.Tensor(1))
        else:
            self.acti_log2_t = torch.Tensor(1)

    def FUNCTION_forward(self, input):
        return qunsigned(F.FUCNTION(input), self.acti_log2_t, self.acti_bit_width)

    def FUNCTION_forward_unquant(self, input):
        return F.FUNCTION(input)

    def quantilize(self):
        self.quant = True

    def floatilize(self):
        self.quant = False

    def forward(self, input):
        return self.FUCNTION_forward(
            input) if self.quant else self.FUNCTION_forward_unquant(input)

```