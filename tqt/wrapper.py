from . import function as tqt
import torch.nn as nn

OFUNCTION = {
    'ReLU':
    lambda inplace=False, acti_bit_width=8, retrain=True: nn.ReLU(inplace=
                                                                  inplace),
    'ReLU6':
    lambda inplace=False, acti_bit_width=8, retrain=True: nn.ReLU6(inplace=
                                                                   inplace),
    'BatchNorm2d':
    lambda num_features, eps=1e-5, momentum=0.1, affine=True,
    track_running_stats=True, weight_bit_width
    =8, bias_bit_width=16, retrain=True: nn.BatchNorm2d(
        num_features, eps, momentum, affine, track_running_stats),
    'Conv2d':
    lambda in_channels, out_channels, kernel_size, stride=1, padding=0,
    dilation=1, groups=1, bias=True, padding_mode='zeros', weight_bit_width=8,
    bias_bit_width=16, retrain=True: nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               dilation=dilation,
                                               groups=groups,
                                               bias=bias,
                                               padding_mode=padding_mode),
    'Linear':
    lambda in_features, out_features, bias=True, weight_bit_width=8,
    bias_bit_width=16, retrain=True: nn.Linear(
        in_features, out_features, bias=bias),
    'MaxPool2d':
    lambda kernel_size, stride=None, padding=0, dilation=1, return_indices=
    False, ceil_mode=False: nn.MaxPool2d(kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         return_indices=return_indices,
                                         ceil_mode=ceil_mode),
    'Adder2d':
    lambda input_channel, output_channel, kernel_size, stride=1, padding=0,
    bias=False, weight_bit_width=8, bias_bit_width=16, retrain=True: tqt.extra.
    Adder2d(input_channel,
            output_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
}

QFUNCTION = {
    'ReLU':
    lambda inplace=False, acti_bit_width=8, retrain=True, quant=False: tqt.
    ReLU(inplace=inplace,
         acti_bit_width=acti_bit_width,
         retrain=retrain,
         quant=quant),
    'ReLU6':
    lambda inplace=False, acti_bit_width=8, retrain=True, quant=False: tqt.
    ReLU6(inplace=inplace,
          acti_bit_width=acti_bit_width,
          retrain=retrain,
          quant=quant),
    'BatchNorm2d':
    lambda num_features, eps=1e-5, momentum=0.1, affine=True,
    track_running_stats=True, weight_bit_width=8, bias_bit_width=16, retrain=
    True, quant=False: tqt.BatchNorm2d(num_features,
                                       eps,
                                       momentum,
                                       affine,
                                       track_running_stats,
                                       weight_bit_width=weight_bit_width,
                                       bias_bit_width=bias_bit_width,
                                       retrain=retrain,
                                       quant=quant),
    'Conv2d':
    lambda in_channels, out_channels, kernel_size, stride=1, padding=0,
    dilation=1, groups=1, bias=True, padding_mode='zeros', weight_bit_width=8,
    bias_bit_width=16, retrain=True, quant=False: tqt.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        weight_bit_width=weight_bit_width,
        bias_bit_width=bias_bit_width,
        retrain=retrain,
        quant=quant),
    'Linear':
    lambda in_features, out_features, bias=True, weight_bit_width=8,
    bias_bit_width=16, retrain=True, quant=False: tqt.Linear(
        in_features,
        out_features,
        bias=bias,
        weight_bit_width=weight_bit_width,
        bias_bit_width=bias_bit_width,
        retrain=retrain,
        quant=quant),
    'MaxPool2d':
    lambda kernel_size, stride=None, padding=0, dilation=1, return_indices=
    False, ceil_mode=False, quant=False: nn.MaxPool2d(kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      dilation=dilation,
                                                      return_indices=
                                                      return_indices,
                                                      ceil_mode=ceil_mode),
    'Adder2d':
    lambda input_channel, output_channel, kernel_size, stride=1, padding=0,
    bias=False, weight_bit_width=8, bias_bit_width=16, retrain=True, quant=
    False: tqt.Adder2d(input_channel,
                       output_channel,
                       kernel_size,
                       stride=stride,
                       padding=padding,
                       bias=bias,
                       weight_bit_width=weight_bit_width,
                       bias_bit_width=bias_bit_width,
                       retrain=retrain,
                       quant=False)
}
