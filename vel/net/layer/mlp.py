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
from vel.net.layer_base import LayerFactory, Layer, LayerInfo, LayerFactoryContext


class MLP(Layer):
    """ Simple Multi-Layer-Perceptron network """
    def __init__(self, info: LayerInfo, input_shape: SizeHints, hidden_layers: typing.List[int],
                 activation: str = 'tanh', normalization: typing.Optional[str] = None):
        super().__init__(info)

        self.input_shape = input_shape
        self.input_length = input_shape.assert_single().last()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.normalization = normalization

        layer_objects = []
        layer_sizes = zip([self.input_length] + hidden_layers, hidden_layers)

        for i_size, o_size in layer_sizes:
            layer_objects.append(nn.Linear(i_size, o_size))

            if self.normalization:
                layer_objects.append(net_util.normalization(normalization)(o_size))

            layer_objects.append(net_util.activation(activation)())

        self.model = nn.Sequential(*layer_objects)
        self.hidden_units = hidden_layers[-1] if hidden_layers else self.input_length

        self.output_shape = SizeHints(input_shape.assert_single().drop_last().append(self.hidden_units))

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
        return self.output_shape


class MLPFactory(LayerFactory):
    def __init__(self, hidden_layers: typing.List[int], activation: str = 'tanh',
                 normalization: typing.Optional[str] = None):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.normalization = normalization

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "mlp"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        return MLP(
            info=self.make_info(context),
            input_shape=direct_input,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            normalization=self.normalization
        )


def create(hidden_layers: [int], activation='tanh', normalization=None, label=None, group=None):
    """ Vel factory function """
    return MLPFactory(
        hidden_layers=hidden_layers, activation=activation, normalization=normalization
    ).with_given_name(label).with_given_group(group)
