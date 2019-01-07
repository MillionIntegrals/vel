"""
Code based loosely on implementation:
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

Under MIT license.
"""
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from vel.api import SupervisedModel, ModelFactory
from vel.metrics.loss_metric import Loss
from vel.metrics.accuracy import Accuracy


class Net(SupervisedModel):
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

    @staticmethod
    def _weight_initializer(tensor):
        init.xavier_uniform_(tensor.weight, gain=init.calculate_gain('relu'))
        init.constant_(tensor.bias, 0.0)

    def __init__(self, img_rows, img_cols, img_channels, num_classes):
        super(Net, self).__init__()

        self.flattened_size = (img_rows - 4) // 2 * (img_cols - 4) // 2 * 64

        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))

        self.dropout1 = nn.Dropout2d(p=0.25)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def reset_weights(self):
        self._weight_initializer(self.conv1)
        self._weight_initializer(self.conv2)
        self._weight_initializer(self.fc1)
        self._weight_initializer(self.fc2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.dropout1(x)
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]


def create(img_rows, img_cols, img_channels, num_classes):
    """ Vel factory function """
    def instantiate(**_):
        return Net(img_rows, img_cols, img_channels, num_classes)

    return ModelFactory.generic(instantiate)
