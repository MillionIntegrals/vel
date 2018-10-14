"""
Code based loosely on the implementation:
https://github.com/openai/baselines/blob/master/baselines/common/models.py

Under MIT license.
"""
import typing
import numpy as np

import torch.nn as nn
import torch.nn.init as init

import vel.util.network as net_util

from vel.api.base import LinearBackboneModel, ModelFactory


class MLP2(LinearBackboneModel):
    """ Simple Multi-Layer-Perceptron network """
    def __init__(self, input_length: int, hidden_layers: typing.List[int], activation: str='tanh',
                 layer_norm: bool=False):
        super().__init__()

        self.input_length = input_length
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.layer_norm = layer_norm

        layer_objects = []
        layer_sizes = zip([input_length] + hidden_layers, hidden_layers)

        for input_size, output_size in layer_sizes:
            layer_objects.append(nn.Linear(input_size, output_size))

            if self.layer_norm:
                layer_objects.append(nn.LayerNorm(output_size))

            layer_objects.append(net_util.activation(activation)())

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


def create(input_length, hidden_layers, activation='tanh', layer_norm=False):
    def instantiate(**_):
        return MLP2(
            input_length=input_length,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=layer_norm
        )

    return ModelFactory.generic(instantiate)


MLPFactory = create
