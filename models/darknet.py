import torch
from torch import nn
from collections import OrderedDict
import math
from torchvision.ops.misc import FrozenBatchNorm2d


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers, norm_layer):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0], norm_layer)
        self.layer2 = self._make_layer([64, 128], layers[1], norm_layer)
        self.layer3 = self._make_layer([128, 256], layers[2], norm_layer)
        self.layer4 = self._make_layer([256, 512], layers[3], norm_layer)
        self.layer5 = self._make_layer([512, 1024], layers[4], norm_layer)

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, norm_layer):
        layers = []
        #  downsample
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", norm_layer(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        #  blocks
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes, norm_layer)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet21(norm_layer=None):
    """Constructs a darknet-21 model.
    """
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    model = DarkNet([1, 1, 2, 2, 1], norm_layer=norm_layer)
    return model


def darknet53(norm_layer=None):
    """Constructs a darknet-53 model.
    """
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    model = DarkNet([1, 2, 8, 8, 4], norm_layer=norm_layer)
    return model

