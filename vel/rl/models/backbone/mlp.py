"""
Code based loosely on implementation:
https://github.com/openai/baselines/blob/master/baselines/common/models.py

Under MIT license.
"""
import numpy as np

import torch.nn as nn
import torch.nn.init as init

import vel.util.network as net_util

from vel.api.base import LinearBackboneModel, ModelFactory


class MLP(LinearBackboneModel):
    """ Simple Multi-Layer-Perceptron network """
    def __init__(self, input_length, layers=2, hidden_units=64, activation='tanh', layer_norm=False):
        super().__init__()

        self.input_length = input_length
        self.layers = layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.layer_norm = layer_norm

        current_size = self.input_length
        layer_objects = []

        for i in range(self.layers):
            layer_objects.append(nn.Linear(current_size, hidden_units))

            if self.layer_norm:
                layer_objects.append(nn.LayerNorm(hidden_units))

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
        input_data = input_data.float()
        return self.model(input_data)


def create(input_length, layers=2, hidden_units=64, activation='tanh', layer_norm=True):
    def instantiate(**_):
        return MLP(
            input_length=input_length,
            layers=layers,
            hidden_units=hidden_units,
            activation=activation,
            layer_norm=layer_norm
        )

    return ModelFactory.generic(instantiate)


MLPFactory = create
