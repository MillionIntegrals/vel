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

from vel.api import LinearBackboneModel, ModelFactory


class MLP(LinearBackboneModel):
    """ Simple Multi-Layer-Perceptron network """
    def __init__(self, input_length: int, hidden_layers: typing.List[int], activation: str='tanh',
                 normalization: typing.Optional[str]=None):
        super().__init__()

        self.input_length = input_length
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.normalization = normalization

        layer_objects = []
        layer_sizes = zip([input_length] + hidden_layers, hidden_layers)

        for input_size, output_size in layer_sizes:
            layer_objects.append(nn.Linear(input_size, output_size))

            if self.normalization:
                layer_objects.append(net_util.normalization(normalization)(output_size))

            layer_objects.append(net_util.activation(activation)())

        self.model = nn.Sequential(*layer_objects)
        self.hidden_units = hidden_layers[-1] if hidden_layers else input_length

    @property
    def output_dim(self) -> int:
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


def create(input_length, hidden_layers, activation='tanh', normalization=None):
    """ Vel factory function """
    def instantiate(**_):
        return MLP(
            input_length=input_length,
            hidden_layers=hidden_layers,
            activation=activation,
            normalization=normalization
        )

    return ModelFactory.generic(instantiate)


MLPFactory = create
