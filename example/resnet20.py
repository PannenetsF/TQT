# 2020.01.10-Replaced conv with adder
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

from tqt.function import Adder2d as adder2d
import tqt.function as nn
import tqt
import torch
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return adder2d(in_planes,
                   out_planes,
                   kernel_size=3,
                   stride=stride,
                   padding=1,
                   bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.proc = ['conv1', 'bn1', 'relu', 'conv2', 'bn2']
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        # self.proc = ['conv1', 'bn1', 'relu']
        self.proc = [
            'conv1', 'bn1', 'relu', 'layer1', 'layer2', 'layer3', 'avgpool',
            'fc', 'bn2'
        ]
        self.conv1 = nn.Conv2d(3,
                               16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                adder2d(self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(
            block(inplanes=self.inplanes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.size(0), -1)


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


if __name__ == '__main__':
    net = resnet20()
    data = torch.rand(3, 3, 32, 32)
    hd = tqt.threshold.add_hook(net, 'net', tqt.threshold.hook_handler)
    net(data)
    tqt.utils.make_net_quant_or_not(net, quant=True)
    tqt.threshold.get_hook(net, 'net')
    tqt.threshold.init_network(net, net, 'net', show=True)
    # tqt.threshold.remove_hook(hd)
    # tqt.threshold.remove_hookout(net, 'net')
    # tqt.threshold.get_hook(net, 'net')
