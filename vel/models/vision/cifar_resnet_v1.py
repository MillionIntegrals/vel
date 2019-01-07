"""
Code based on
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F

from vel.api import SupervisedModel, ModelFactory
from vel.modules.resnet_v1 import Bottleneck, BasicBlock


class ResNetV1(SupervisedModel):
    """ A ResNet V1 model as defined in the literature """

    def __init__(self, block, layers, inplanes, divisor=4, img_channels=3, num_classes=1000):
        super().__init__()

        self.num_classess = num_classes
        self.inplanes = inplanes
        self.divisor = divisor

        self.pre_conv = nn.Conv2d(img_channels, inplanes, kernel_size=(3, 3), padding=1, bias=False)
        self.pre_bn = nn.BatchNorm2d(inplanes)

        self.layer1 = self._make_layer(block, inplanes,     inplanes,     layers[0], stride=1)
        self.layer2 = self._make_layer(block, inplanes,     inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 2, inplanes * 4, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes * 4, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []

        # First block is the one doing the rescaling
        layers.append(block(in_channels, out_channels, stride, divisor=self.divisor))

        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, divisor=self.divisor))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.pre_bn(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate value of the loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        from vel.metrics.loss_metric import Loss
        from vel.metrics.accuracy import Accuracy
        return [Loss(), Accuracy()]


def create(blocks, mode='basic', inplanes=16, divisor=4, num_classes=1000):
    """ Vel factory function """
    block_dict = {
        'basic': BasicBlock,
        'bottleneck': Bottleneck
    }

    def instantiate(**_):
        return ResNetV1(block_dict[mode], blocks, inplanes=inplanes, divisor=divisor, num_classes=num_classes)

    return ModelFactory.generic(instantiate)
