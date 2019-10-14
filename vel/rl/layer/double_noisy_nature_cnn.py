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

from vel.api import SizeHints, SizeHint
from vel.net.layer_base import Layer, LayerFactory, LayerFactoryContext, LayerInfo
from vel.rl.module.noisy_linear import NoisyLinear


class DoubleNoisyNatureCnn(Layer):
    """
    Neural network as defined in the paper 'Human-level control through deep reinforcement learning'
    but with two separate heads and "noisy" linear layer.
    """
    def __init__(self, info: LayerInfo, input_width, input_height, input_channels, output_dim=512, initial_std_dev=0.4,
                 factorized_noise=True):
        super().__init__(info)

        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(8, 8),
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4),
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1
        )

        self.final_width = net_util.convolutional_layer_series(input_width, [
            (8, 0, 4),
            (4, 0, 2),
            (3, 0, 1)
        ])

        self.final_height = net_util.convolutional_layer_series(input_height, [
            (8, 0, 4),
            (4, 0, 2),
            (3, 0, 1)
        ])

        self.linear_layer_one = NoisyLinear(
            # 64 is the number of channels of the last conv layer
            self.final_width * self.final_height * 64,
            self.output_dim,
            initial_std_dev=initial_std_dev,
            factorized_noise=factorized_noise
        )

        self.linear_layer_two = NoisyLinear(
            # 64 is the number of channels of the last conv layer
            self.final_width * self.final_height * 64,
            self.output_dim,
            initial_std_dev=initial_std_dev,
            factorized_noise=factorized_noise
        )

    def size_hints(self) -> SizeHints:
        return SizeHints((
            SizeHint(None, self.output_dim),
            SizeHint(None, self.output_dim)
        ))

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
            elif isinstance(m, NoisyLinear):
                m.reset_weights()

    def forward(self, image, state: dict = None, context: dict = None):
        if context is not None:
            deterministic = context.get('deterministic', False)
        else:
            deterministic = False

        result = image
        result = F.relu(self.conv1(result))
        result = F.relu(self.conv2(result))
        result = F.relu(self.conv3(result))
        flattened = result.view(result.size(0), -1)

        output_one = F.relu(self.linear_layer_one(flattened, deterministic=deterministic))
        output_two = F.relu(self.linear_layer_two(flattened, deterministic=deterministic))

        return output_one, output_two


class DoubleNoisyNatureCnnFactory(LayerFactory):
    """ Nature Cnn Network Factory """

    def __init__(self, initial_std_dev: float = 0.4, factorized_noise: bool = True, output_dim: int = 512):
        super().__init__()
        self.initial_std_dev = initial_std_dev
        self.factorized_noise = factorized_noise
        self.output_dim = output_dim

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "double_noisy_nature_cnn"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        (b, c, w, h) = direct_input.assert_single(4)

        return DoubleNoisyNatureCnn(
            info=self.make_info(context),
            input_width=w,
            input_height=h,
            input_channels=c,
            output_dim=self.output_dim,
            initial_std_dev=self.initial_std_dev,
            factorized_noise=self.factorized_noise
        )


def create(initial_std_dev: float = 0.4, factorized_noise: bool = True, output_dim: int = 512,
           label=None, group=None):
    """ Vel factory function """
    return DoubleNoisyNatureCnnFactory(
        output_dim=output_dim,
        initial_std_dev=initial_std_dev,
        factorized_noise=factorized_noise
    ).with_given_name(label).with_given_group(group)
