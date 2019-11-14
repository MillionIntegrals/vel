import typing
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from vel.api import SizeHint, SizeHints
from vel.net.layer_base import Layer, LayerFactory, LayerInfo, LayerFactoryContext


class FcResidual(Layer):
    """ Residual fully-connected block """

    def __init__(self, info: LayerInfo, input_shape: SizeHint, divisor: int = 1, activation: str = 'relu',
                 normalization: typing.Optional[str] = None):
        super().__init__(info)

        self._size_hints = SizeHints(input_shape)

        self.trunk_shape = input_shape[-1]
        self.bottleneck_shape = self.trunk_shape // divisor

        self.f1 = nn.Linear(self.trunk_shape, self.bottleneck_shape)

        if normalization == 'layer':
            self.n1 = nn.LayerNorm(self.bottleneck_shape)
        elif normalization is None:
            self.n1 = nn.Identity()
        else:
            raise NotImplementedError

        if activation == 'relu':
            self.a1 = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        self.f2 = nn.Linear(self.bottleneck_shape, self.trunk_shape)

        if normalization == 'layer':
            self.n2 = nn.LayerNorm(self.trunk_shape)
        elif normalization is None:
            self.n2 = nn.Identity()
        else:
            raise NotImplementedError

        if activation == 'relu':
            self.a2 = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self._size_hints

    def forward(self, direct, state: dict, context: dict):
        residual = direct

        residual = self.a1(self.n1(self.f1(residual)))
        residual = self.a2(self.n2(self.f2(residual)))

        return residual + direct


class FcResidualFactory(LayerFactory):
    """ Factory for fully-connected residual layers """
    def __init__(self, divisor: int, activation: str, normalization: typing.Optional[str] = None):
        super().__init__()
        self.divisor = divisor
        self.activation = activation
        self.normalization = normalization

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "fc_residual"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        size_hint = direct_input.assert_single()
        info = self.make_info(context)

        return FcResidual(
            info=info,
            input_shape=size_hint,
            divisor=self.divisor,
            activation=self.activation,
            normalization=self.normalization
        )


def create(divisor: int, activation: str = 'relu', normalization: typing.Optional[str] = None,
           label=None, group=None):
    return FcResidualFactory(divisor, activation, normalization)

