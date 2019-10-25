import itertools as it

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from vel.api import ModuleFactory
from vel.module.layers import Flatten, Reshape

from vel.model.latent.vae_base import VaeBase


class FcVae(VaeBase):
    """
    A simple variational autoencoder, containing 2 fully connected layers.
    """

    def __init__(self, img_rows, img_cols, img_channels, layers=None, representation_length=32,
                 analytical_kl_div=False):
        super().__init__(analytical_kl_div=analytical_kl_div)

        if layers is None:
            layers = [512, 256]

        self.representation_length = representation_length
        self.layers = layers

        input_length = img_rows * img_cols * img_channels

        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=input_length, out_features=self.layers[0]),
            nn.Tanh(),
            nn.Linear(in_features=self.layers[0], out_features=self.layers[1]),
            nn.Tanh(),
            nn.Linear(self.layers[1], representation_length * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=representation_length, out_features=self.layers[1]),
            nn.Tanh(),
            nn.Linear(in_features=self.layers[1], out_features=self.layers[0]),
            nn.Tanh(),
            nn.Linear(in_features=self.layers[0], out_features=input_length),
            Reshape(img_channels, img_rows, img_cols),
            nn.Sigmoid()
        )

        self.register_buffer('prior_mean', torch.tensor([[0.0]]))
        self.register_buffer('prior_std', torch.tensor([[1.0]]))

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

    # @staticmethod
    # def _weight_initializer(tensor):
    #     init.xavier_normal_(tensor.weight, gain=init.calculate_gain('tanh'))
    #     init.zeros_(tensor.bias)
    #
    # def reset_weights(self):
    #     for m in it.chain(self.encoder.modules(), self.decoder.modules()):
    #         if isinstance(m, nn.Conv2d):
    #             self._weight_initializer(m)
    #         elif isinstance(m, nn.ConvTranspose2d):
    #             self._weight_initializer(m)
    #         elif isinstance(m, nn.Linear):
    #             self._weight_initializer(m)


def create(img_rows, img_cols, img_channels, layers=None, representation_length=32,
           analytical_kl_div=True):
    """ Vel factory function """
    if layers is None:
        layers = [512, 256]

    def instantiate(**_):
        return FcVae(
            img_rows, img_cols, img_channels, layers=layers, representation_length=representation_length,
            analytical_kl_div=analytical_kl_div
        )

    return ModuleFactory.generic(instantiate)
