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
# qnet.load_state_dict(net.state_dict(), strict=False)
# x = torch.rand(50, 1, 28, 28)
# handler = tqt.threshold.hook_handler
# handles = tqt.threshold.add_hook_general(net.proc, handler)
# net(x)
# tqt.threshold.init.init_network(net.proc, qnet.proc, show=True)
# tqt.threshold.remove_hook(handles)
# torch.save(qnet.state_dict(), 'save_qnet.pth')
# q_state = torch.load('save_qnet.pth')
# new_qnet = lenet(quant=True)
# new_qnet.load_state_dict(q_state)