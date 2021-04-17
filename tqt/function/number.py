"""
Two basic data types are defined for forward propagation and back propagation.
Signed and unsigned fixed point data needed Straight-Through filter for back propagation.
"""

import torch
from torch.autograd import Function
import math
from ._utils import *


class Ceil(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.ceil(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return torch.ones(grad_output.shape).type_as(grad_output)


ceil = Ceil.apply


class RoundToEven(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.ones(grad_output.shape).type_as(grad_output)


round = RoundToEven.apply


class qSigned(Function):
    @staticmethod
    def forward(ctx, x, log2_t, bit_width):
        r'''
        See TQT paper's Eqn. (4)
        '''
        bit_max = 2.**(bit_width - 1)
        n = -bit_max
        p = bit_max - 1
        s = 2.**ceil(log2_t) / bit_max
        q = torch.clamp(round(x / s), n, p) * s
        ctx.save_for_backward(x / s, s, number_to_tensor(n, x),
                              number_to_tensor(p, x))
        return q

    @staticmethod
    def backward(ctx, grad_output):
        x_div_s, s, n, p = ctx.saved_tensors
        rounded = round(x_div_s)
        cmp0 = (n <= rounded) & (rounded <= p)
        cmp1 = rounded < n
        cmp2 = rounded > p
        grad_s = (rounded - x_div_s) * cmp0 + n * cmp1 + p * cmp2
        grad_log_2_t = math.log(2) * grad_s * s
        grad_x = cmp0 * 1.0
        return grad_output * grad_x, grad_output * grad_log_2_t, None


qsigned = qSigned.apply


class qUnsigned(Function):
    @staticmethod
    def forward(ctx, x, log2_t, bit_width):
        r'''
        See TQT paper's Eqn. (4)
        '''
        bit_max = 2.**(bit_width)
        n = 0
        p = bit_max - 1
        s = 2.**ceil(log2_t) / bit_max
        q = torch.clamp(round(x / s), n, p) * s
        ctx.save_for_backward(round(x / s), s, number_to_tensor(n, x),
                              number_to_tensor(p, x))
        return q

    @staticmethod
    def backward(ctx, grad_output):
        x_div_s, s, n, p = ctx.saved_tensors
        rounded = round(x_div_s)
        cmp0 = (n <= rounded) & (rounded <= p)
        cmp1 = rounded < n
        cmp2 = rounded > p
        grad_s = (rounded - x_div_s) * cmp0 + n * cmp1 + p * cmp2
        grad_log_2_t = math.log(2) * grad_s * s
        grad_x = cmp0 * 1.0
        return grad_output * grad_x, grad_output * grad_log_2_t, None


qunsigned = qUnsigned.apply