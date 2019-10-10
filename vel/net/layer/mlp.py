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

from vel.api import SizeHints
from vel.net.layer_base import LayerFactory, Layer


class MLP(Layer):
    """ Simple Multi-Layer-Perceptron network """
    def __init__(self, name: str, input_size: SizeHints, hidden_layers: typing.List[int], activation: str = 'tanh',
                 normalization: typing.Optional[str] = None):
        super().__init__(name)

        self.input_size = input_size
        self.input_length = input_size.assert_single().last()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.normalization = normalization

        layer_objects = []
        layer_sizes = zip([self.input_length] + hidden_layers, hidden_layers)

        for input_size, output_size in layer_sizes:
            layer_objects.append(nn.Linear(input_size, output_size))

            if self.normalization:
                layer_objects.append(net_util.normalization(normalization)(output_size))

            layer_objects.append(net_util.activation(activation)())

        self.model = nn.Sequential(*layer_objects)
        self.hidden_units = hidden_layers[-1] if hidden_layers else self.input_length

        self.output_size = input_size.assert_single().drop_last().append(self.hidden_units)

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, direct, state: dict = None, context: dict = None):
        return self.model(direct.float())

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return SizeHints(self.output_size)


class MLPFactory(LayerFactory):
    def __init__(self, hidden_layers: typing.List[int], activation: str = 'tanh',
                 normalization: typing.Optional[str] = None):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.normalization = normalization

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "mlp"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict, extra_args: dict) -> Layer:
        """ Create a given layer object """
        return MLP(
            name=name,
            input_size=direct_input,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            normalization=self.normalization
        )


def create(hidden_layers, activation='tanh', normalization=None):
    """ Vel factory function """
    return MLPFactory(hidden_layers=hidden_layers, activation=activation, normalization=normalization)
