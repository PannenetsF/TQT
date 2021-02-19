import tqt
import torch.nn as nn
import torch


class lenet(nn.Module):
    def __init__(self, quant=False):
        super().__init__()
        self.proc = nn.ModuleList()
        dic = tqt.wrapper.OFUNCTION if quant is False else tqt.wrapper.QFUNCTION
        self.proc.append(dic['Conv2d'](1, 6, 5, 1, 2))
        self.proc.append(dic['MaxPool2d']((2, 2)))
        self.proc.append(dic['Conv2d'](6, 16, 5))
        self.proc.append(dic['MaxPool2d']((2, 2)))
        self.proc.append(dic['Conv2d'](16, 120, 5))
        self.proc.append(dic['Linear'](120, 84))
        self.proc.append(dic['Linear'](84, 10))

    def forward(self, x):
        for idx, p in enumerate(self.proc):
            x = p(x)
            if (idx == 4):
                x = x.reshape(x.shape[0], -1)
        return x


net = lenet(True)
x = torch.rand(50, 1, 28, 28)
print(net(x).shape)
