import itertools as it

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributions as dist

import vel.util.network as net_util

from vel.api import ModelFactory
from vel.module.layers import Flatten, Reshape
from vel.model.latent.iwae import IWAE


class CnnIWAE(IWAE):
    """
    A simple IWAE, containing 3 convolutional layers
    """

    def __init__(self, img_rows, img_cols, img_channels, k=5, channels=None, representation_length=32,
                 analytical_kl_div=True):
        super().__init__(k=k, analytical_kl_div=analytical_kl_div)

        if channels is None:
            channels = [16, 32, 32]

        layer_series = [
            (3, 1, 1),
            (3, 1, 2),
            (3, 1, 2),
        ]

        self.representation_length = representation_length

        self.final_width = net_util.convolutional_layer_series(img_rows, layer_series)
        self.final_height = net_util.convolutional_layer_series(img_cols, layer_series)
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=channels[0], kernel_size=(3, 3), padding=1),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.LayerNorm([
                channels[0],
                net_util.convolutional_layer_series(img_rows, layer_series[:1]),
                net_util.convolutional_layer_series(img_cols, layer_series[:1]),
            ]),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3, 3), stride=2, padding=1),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.LayerNorm([
                channels[1],
                net_util.convolutional_layer_series(img_rows, layer_series[:2]),
                net_util.convolutional_layer_series(img_cols, layer_series[:2]),
            ]),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3), stride=2, padding=1),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.LayerNorm([
                channels[2],
                net_util.convolutional_layer_series(img_rows, layer_series),
                net_util.convolutional_layer_series(img_cols, layer_series),
            ]),
            Flatten(),
            nn.Linear(self.final_width * self.final_height * channels[2], representation_length * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(representation_length, self.final_width * self.final_height * channels[2]),
            # nn.ReLU(True),
            nn.SELU(True),
            Reshape(channels[2], self.final_width, self.final_height),
            nn.LayerNorm([
                channels[2],
                net_util.convolutional_layer_series(img_rows, layer_series),
                net_util.convolutional_layer_series(img_cols, layer_series),
            ]),
            nn.ConvTranspose2d(
                in_channels=channels[2], out_channels=channels[1], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.LayerNorm([
                channels[1],
                net_util.convolutional_layer_series(img_rows, layer_series[:2]),
                net_util.convolutional_layer_series(img_cols, layer_series[:2]),
            ]),
            nn.ConvTranspose2d(
                in_channels=channels[1], out_channels=channels[0], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.LayerNorm([
                channels[0],
                net_util.convolutional_layer_series(img_rows, layer_series[:1]),
                net_util.convolutional_layer_series(img_cols, layer_series[:1]),
            ]),
            nn.ConvTranspose2d(in_channels=channels[0], out_channels=img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.register_buffer('prior_mean', torch.tensor([[0.0]]))
        self.register_buffer('prior_std', torch.tensor([[1.0]]))

    @staticmethod
    def _weight_initializer(tensor):
        init.xavier_uniform_(tensor.weight, gain=init.calculate_gain('relu'))
        init.constant_(tensor.bias, 0.0)

    def reset_weights(self):
        for m in it.chain(self.encoder, self.decoder):
            if isinstance(m, nn.Conv2d):
                self._weight_initializer(m)
            elif isinstance(m, nn.ConvTranspose2d):
                self._weight_initializer(m)
            elif isinstance(m, nn.Linear):
                self._weight_initializer(m)

    def encoder_network(self, sample: torch.Tensor) -> torch.Tensor:
        """ Transform input sample into an encoded representation """
        return self.encoder(sample)

    def encoder_distribution(self, encoded: torch.Tensor) -> dist.Distribution:
        """ Create a pytorch distribution object representing the encoder distribution (approximate posterior) """
        mu = encoded[:, :self.representation_length]
        std = F.softplus(encoded[:, self.representation_length:])

        return dist.Independent(dist.Normal(mu, std), 1)

    def decoder_network(self, z: torch.Tensor) -> torch.Tensor:
        """ Transform encoded value into a decoded representation """
        return self.decoder(z)

    def decoder_distribution(self, decoded: torch.Tensor) -> dist.Distribution:
        """ Create a pytorch distribution object representing the decoder distribution (likelihood) """
        return dist.Independent(dist.Bernoulli(probs=decoded), 3)

    def prior_distribution(self) -> dist.Distribution:
        """ Return a prior distribution object """
        return dist.Independent(dist.Normal(self.prior_mean, self.prior_std), 1)

    def decoder_sample(self, decoded: torch.Tensor) -> torch.Tensor:
        """ Sample from a decoder distribution - we ignore that since it's so weak in this case """
        return decoded


def create(img_rows, img_cols, img_channels, k=5, channels=None, representation_length=32):
    """ Vel factory function """
    if channels is None:
        channels = [16, 32, 32]

    def instantiate(**_):
        return CnnIWAE(
            img_rows, img_cols, img_channels, k=k, channels=channels, representation_length=representation_length
        )

    return ModelFactory.generic(instantiate)
