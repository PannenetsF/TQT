import torch
import tqt
import tqt.function as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.mp1 = nn.MaxPool2d((2, 2))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.mp2 = nn.MaxPool2d((2, 2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        self.proc = [
            'conv1', 'mp1', 'relu1', 'conv2', 'mp2', 'relu2', 'conv3', 'relu3',
            'fc1', 'relu4', 'fc2'
        ]

    def forward(self, x):
        for p in self.proc:
            x = getattr(self, p)(x)
            if p == 'conv3':
                x = x.reshape(x.shape[0], -1)
        return x


if __name__ == '__main__':
    x = torch.rand(3, 1, 28, 28)
    net = LeNet()
    tqt.utils.make_net_quant_or_not(net,
                                    'net',
                                    quant=True,
                                    exclude=[torch.nn.ReLU],
                                    show=True)
    tqt.threshold.init.init_network(net, net, 'net', show=True)
    print(net(x).shape)
