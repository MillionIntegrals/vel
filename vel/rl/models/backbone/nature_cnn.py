"""
Code based loosely on implementation:
https://github.com/openai/baselines/blob/master/baselines/ppo2/policies.py

Under MIT license.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import vel.util.network as net_util

from vel.api.base import LinearBackboneModel, ModelFactory


class NatureCnn(LinearBackboneModel):
    """ Neural network as defined in the paper 'Human-level control through deep reinforcement learning'"""
    def __init__(self, input_width, input_height, input_channels, output_dim=512,
                 kernel1=8, kernel2=4, kernel3=3):
        super().__init__()
        self._output_dim = output_dim
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=self.kernel1,
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=self.kernel2,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=self.kernel3,
            stride=1
        )

        self.final_width = net_util.convolutional_layer_series(input_width, [
            (self.kernel1, 0, 4),
            (self.kernel2, 0, 2),
            (self.kernel3, 0, 1)
        ])

        self.final_height = net_util.convolutional_layer_series(input_height, [
            (self.kernel1, 0, 4),
            (self.kernel2, 0, 2),
            (self.kernel3, 0, 1)
        ])

        self.linear_layer = nn.Linear(
            self.final_width * self.final_height * 64,  # 64 is the number of channels of the last conv layer
            self.output_dim
        )

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        """ Call proper initializers for the weights """
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
        result = image.permute(0, 3, 1, 2).contiguous().type(torch.float) / 255.0
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        result = F.relu(self.conv3(result))
        flattened = result.view(result.size(0), -1)
        return F.relu(self.linear_layer(flattened))


def create(input_width, input_height, input_channels=1, output_dim=512, kernel1=8, kernel2=4, kernel3=3):
    def instantiate(**_):
        return NatureCnn(
            input_width=input_width, input_height=input_height, input_channels=input_channels,
            output_dim=output_dim, kernel1=kernel1, kernel2=kernel2, kernel3=kernel3 
        )

    return ModelFactory.generic(instantiate)


# Add this to make nicer scripting interface
NatureCnnFactory = create

