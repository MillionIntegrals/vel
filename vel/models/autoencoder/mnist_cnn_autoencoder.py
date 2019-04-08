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

    def __init__(self, img_rows, img_cols, img_channels, num_classes):
        super(MnistCnnAutoencoder, self).__init__()

        self.flattened_size = (img_rows - 4) // 2 * (img_cols - 4) // 2 * 64

        layer_series = [
            (3, 1, 1),
            (3, 1, 2),
            (3, 1, 2),
        ]

        self.final_width = net_util.convolutional_layer_series(img_rows, layer_series)
        self.final_height = net_util.convolutional_layer_series(img_cols, layer_series)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            Flatten(),
            nn.Linear(self.final_width * self.final_height * 32, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, self.final_width * self.final_height * 32),
            nn.ReLU(True),
            Reshape(32, self.final_width, self.final_height),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=img_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def _weight_initializer(tensor):
        init.xavier_uniform_(tensor.weight, gain=init.calculate_gain('relu'))
        init.constant_(tensor.bias, 0.0)

    def reset_weights(self):
        for m in self.children():
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


def create(img_rows, img_cols, img_channels, num_classes):
    """ Vel factory function """
    def instantiate(**_):
        return MnistCnnAutoencoder(img_rows, img_cols, img_channels, num_classes)

    return ModelFactory.generic(instantiate)
