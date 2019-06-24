import torch
import torch.nn as nn

import vel.util.module_util as mu

from vel.metric.loss_metric import Loss
from vel.util.summary import summary


class Model(nn.Module):
    """ Class representing full neural network model """

    def metrics(self) -> list:
        """ Set of metrics for this model """
        return []

    def train(self, mode=True):
        r"""
        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super().train(mode)

        if mode:
            mu.apply_leaf(self, mu.set_train_mode)

        return self

    def summary(self, input_size=None):
        """ Print a model summary """

        if input_size is None:
            print(self)
            print("-" * 100)
            number = sum(p.numel() for p in self.parameters())
            print("Number of model parameters: {:,}".format(number))
            print("-" * 100)
        else:
            summary(self, input_size)

    def get_layer_groups(self):
        """ Return layers grouped for optimization purposes """
        return [self]

    def reset_weights(self):
        """ Call proper initializers for the weights """
        pass

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return False

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return None


class GradientModel(Model):
    """ Model for a supervised learning problem """

    def calculate_gradient(self, data: dict) -> dict:
        """
        Calculate gradient for given batch of supervised learning.
        Returns a dictionary of metrics
        """
        raise NotImplementedError


class LossFunctionModel(GradientModel):
    """ Model for a supervised learning with a simple loss function """

    def metrics(self) -> list:
        """ Set of metrics for this model """
        return [Loss()]

    def calculate_gradient(self, data: dict) -> dict:
        y_hat = self(data['x'])
        loss_value = self.loss_value(data['x'], data['y'], y_hat)

        if self.training:
            loss_value.backward()

        return {
            'loss': loss_value.item(),
            'data': data['x'],
            'target': data['y'],
            'output': y_hat
        }

    def loss_value(self, x_data, y_true, y_pred) -> torch.tensor:
        """ Calculate a value of loss function """
        raise NotImplementedError


class BackboneModel(Model):
    """ Model that serves as a backbone network to connect your heads to """


class LinearBackboneModel(BackboneModel):
    """
    Model that serves as a backbone network to connect your heads to.
    Has a final output of a single-dimensional linear layer.
    """

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        raise NotImplementedError
