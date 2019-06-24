import itertools as it

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import vel.util.network as net_util

from vel.api import GradientModel, ModelFactory
from vel.metric.averaging_metric import AveragingNamedMetric
from vel.metric.loss_metric import Loss
from vel.module.layers import Flatten, Reshape


class MnistCnnVAE(GradientModel):
    """
    A simple MNIST variational autoencoder, containing 3 convolutional layers.
    """

    def __init__(self, img_rows, img_cols, img_channels, channels=None, representation_length=32):
        super(MnistCnnVAE, self).__init__()

        assert representation_length % 2 == 0, "Representation length must be even"

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
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3), stride=2, padding=1),
            Flatten(),
            nn.Linear(self.final_width * self.final_height * channels[2], representation_length * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(representation_length, self.final_width * self.final_height * channels[2]),
            nn.ReLU(True),
            Reshape(channels[2], self.final_width, self.final_height),
            nn.ConvTranspose2d(
                in_channels=channels[2], out_channels=channels[1], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=channels[1], out_channels=channels[0], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=channels[0], out_channels=img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

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

    def encode(self, sample):
        encoding = self.encoder(sample)

        mu = encoding[:, :self.representation_length]
        # I encode std directly as a softplus, rather than exp(logstd)
        std = F.softplus(encoding[:, self.representation_length:])

        return mu + torch.randn_like(std) * std

    def decode(self, sample):
        return self.decoder(sample)

    def forward(self, sample):
        encoding = self.encoder(sample)

        mu = encoding[:, :self.representation_length]
        # I encode std directly as a softplus, rather than exp(logstd)
        std = F.softplus(encoding[:, self.representation_length:])

        z = mu + torch.randn_like(std) * std

        decoded = self.decoder(z)

        return {
            'decoded': decoded,
            'encoding': z,
            'mu': mu,
            'std': std
        }

    def calculate_gradient(self, x_data, y_true):
        """ Calculate a gradient of loss function """
        output = self(x_data)

        y_pred = output['decoded']

        mu = output['mu']
        std = output['std']
        var = std ** 2

        kl_divergence = - 0.5 * (1 + torch.log(var) - mu ** 2 - var).sum(dim=1)
        kl_divergence = kl_divergence.mean()

        # reconstruction = 0.5 * F.mse_loss(y_pred, y_true)

        # We must sum over all image axis and average only on minibatch axis
        reconstruction = F.binary_cross_entropy(y_pred, y_true, reduce=False).sum(1).sum(1).sum(1).mean()
        loss = reconstruction + kl_divergence

        if self.training:
            loss.backward()

        return {
            'loss': loss.item(),
            'reconstruction': reconstruction.item(),
            'kl_divergence': kl_divergence.item()
        }

    def metrics(self):
        """ Set of metrics for this model """
        return [
            Loss(),
            AveragingNamedMetric('reconstruction'),
            AveragingNamedMetric('kl_divergence')
        ]


def create(img_rows, img_cols, img_channels, channels=None, representation_length=32):
    """ Vel factory function """
    if channels is None:
        channels = [16, 32, 32]

    def instantiate(**_):
        return MnistCnnVAE(
            img_rows, img_cols, img_channels, channels=channels, representation_length=representation_length
        )

    return ModelFactory.generic(instantiate)
