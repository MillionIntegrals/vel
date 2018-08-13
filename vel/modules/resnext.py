"""
Code based on
https://github.com/fastai/fastai/blob/master/fastai/models/cifar10/resnext.py
"""
import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, cardinality, divisor, stride=1):
        super(ResNeXtBottleneck, self).__init__()

        self.cardinality = cardinality
        self.stride = stride
        self.divisor = divisor

        # D is a size of a single group
        # Intermediate layers have D * C channels
        D = out_channels // divisor
        C = cardinality

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.conv_reduce = nn.Conv2d(in_channels, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D*C)

        self.conv_conv = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D*C)

        self.conv_expand = nn.Conv2d(D * C, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x

        return F.relu(residual + bottleneck, inplace=True)
