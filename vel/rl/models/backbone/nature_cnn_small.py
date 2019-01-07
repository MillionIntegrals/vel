"""
Code based loosely on implementation:
https://github.com/openai/baselines/blob/master/baselines/ppo2/policies.py

Under MIT license.
"""
import numpy as np

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import vel.util.network as net_util

from vel.api import LinearBackboneModel, ModelFactory


class NatureCnnSmall(LinearBackboneModel):
    """
    Neural network as defined in the paper 'Human-level control through deep reinforcement learning'
    Smaller version.
    """
    def __init__(self, input_width, input_height, input_channels, output_dim=128):
        super().__init__()

        self._output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=8,
            kernel_size=(8, 8),
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(4, 4),
            stride=2
        )

        self.final_width = net_util.convolutional_layer_series(input_width, [
            (8, 0, 4),
            (4, 0, 2),
        ])

        self.final_height = net_util.convolutional_layer_series(input_height, [
            (8, 0, 4),
            (4, 0, 2),
        ])

        self.linear_layer = nn.Linear(
            self.final_width * self.final_height * 16,
            self.output_dim
        )

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, image):
        result = image
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        flattened = result.view(result.size(0), -1)
        return F.relu(self.linear_layer(flattened))


def create(input_width, input_height, input_channels=1):
    """ Vel factory function """
    def instantiate(**_):
        return NatureCnnSmall(input_width=input_width, input_height=input_height, input_channels=input_channels)

    return ModelFactory.generic(instantiate)


NatureCnnSmallFactory = create
