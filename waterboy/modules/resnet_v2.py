"""
Code based on
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3 convolution with padding.
    Original code has had bias turned off, because Batch Norm would remove the bias either way
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Single residual block consisting of two convolutional layers and a nonlinearity between them
    """
    def __init__(self, in_channels, out_channels, stride=1, divisor=None):
        super().__init__()

        self.stride = stride
        self.divisor = divisor

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)

        # If we need to rescale the "main path"
        if self.shortcut:
            residual = self.shortcut(out)
        else:
            residual = x

        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class Bottleneck(nn.Module):
    """
    A 'bottleneck' residual block consisting of three convolutional layers, where the first one is a downsampler,
    then we have a 3x3 followed by an upsampler.
    """
    def __init__(self, in_channels, out_channels, stride=1, divisor=4):
        super().__init__()

        self.stride = stride
        self.divisor = divisor

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = None

        self.bottleneck_channels = out_channels // divisor

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.conv2 = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(self.bottleneck_channels)
        self.conv3 = nn.Conv2d(self.bottleneck_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # If we need to rescale the "main path"

        if self.shortcut:
            residual = self.shortcut(out)
        else:
            residual = x

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))

        out += residual

        return out
