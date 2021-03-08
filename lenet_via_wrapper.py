import tqt
import torch.nn as nn
import torch


class lenet(nn.Module):
    def __init__(self, quant=False):
        super().__init__()
        self.proc = nn.ModuleList()
        dic = tqt.wrapper.QFUNCTION
        self.proc.append(dic['Conv2d'](1, 6, 5, 1, 2, bias=False, quant=quant))
        self.proc.append(dic['ReLU'](quant=quant))
        self.proc.append(dic['MaxPool2d']((2, 2), quant=quant))
        self.proc.append(dic['Conv2d'](6, 16, 5, quant=quant))
        self.proc.append(dic['ReLU'](quant=quant))
        self.proc.append(dic['MaxPool2d']((2, 2), quant=quant))
        self.proc.append(dic['Conv2d'](16, 120, 5, quant=quant))
        self.proc.append(dic['ReLU'](quant=quant))
        self.proc.append(dic['Linear'](120, 84, quant=quant))
        self.proc.append(dic['ReLU'](quant=quant))
        self.proc.append(dic['Linear'](84, 10, quant=quant))

    def forward(self, x):
        for idx, p in enumerate(self.proc):
            x = p(x)
            if (idx == 7):
                x = x.reshape(x.shape[0], -1)
        return x


net = lenet(quant=False)
qnet = lenet(quant=True)
tqt.utils.make_net_quant_or_not(qnet, quant=True)