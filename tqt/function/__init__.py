from torch.nn import Module, ModuleList, ModuleDict, Sequential

from torch.nn import Parameter

# from torch import init
# from torch import utils

from torch.nn import MaxPool1d, MaxPool2d, MaxPool3d
from torch.nn import AvgPool1d, AvgPool2d, AvgPool3d

from .activation import ReLU, ReLU6
from .batchnorm import BatchNorm2d
from .conv import Conv2d
from .linear import Linear

from . import extra
from .adder import Adder2d

from .number import qsigned, qunsigned