"""
Code based loosely on implementation:
https://github.com/openai/baselines/blob/master/baselines/common/models.py

Under MIT license.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import vel.util.network as net_util

from vel.api.base import LinearBackboneModel, ModelFactory


class MLP(LinearBackboneModel):
    """ Simple Multi-Layer-Perceptron network """
    def __init__(self, input_length, layers=2, hidden_units=64, activation='tanh'):
        super().__init__()

        self.input_length = input_length
        self.layers = layers
        self.hidden_units = hidden_units
        self.activation = activation

        current_size = self.input_length
        layer_objects = []

        for i in range(self.layers):
            layer_objects.append(nn.Linear(current_size, hidden_units))
            layer_objects.append(net_util.activation(activation)())

            current_size = hidden_units

        self.model = nn.Sequential(*layer_objects)

    @property
    def output_dim(self):
        """ Final dimension of model output """
        return self.hidden_units

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, input_data):
        return self.model(input_data)


def create(input_length, layers=2, hidden_units=64, activation='tanh'):
    def instantiate(**_):
        return MLP(
            input_length=input_length,
            layers=layers,
            hidden_units=hidden_units,
            activation=activation
        )

    return ModelFactory.generic(instantiate)
