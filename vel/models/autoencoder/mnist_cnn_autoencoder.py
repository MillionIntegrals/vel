import itertools as it

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import vel.util.network as net_util

from vel.api import SupervisedModel, ModelFactory
from vel.metrics.loss_metric import Loss
from vel.modules.layers import Flatten, Reshape


class MnistCnnAutoencoder(SupervisedModel):
    """
    A simple MNIST classification model.

    Conv 3x3 - 32
    Conv 3x3 - 64
    MaxPool 2x2
    Dropout 0.25
    Flatten
    Dense - 128
    Dense - output (softmax)
    """

    def __init__(self, img_rows, img_cols, img_channels, channels=None, representation_length=32):
        super(MnistCnnAutoencoder, self).__init__()

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
            nn.Linear(self.final_width * self.final_height * channels[2], representation_length)
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

    def forward(self, x):
        encoding = self.encoder(x)
        decoded = self.decoder(encoding)
        return decoded

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        return F.mse_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss()]


def create(img_rows, img_cols, img_channels, channels=None, representation_length=32):
    """ Vel factory function """
    if channels is None:
        channels = [16, 32, 32]

    def instantiate(**_):
        return MnistCnnAutoencoder(
            img_rows, img_cols, img_channels, channels=channels, representation_length=representation_length
        )

    return ModelFactory.generic(instantiate)
