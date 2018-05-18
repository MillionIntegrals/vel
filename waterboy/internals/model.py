import torch.nn as nn

from waterboy.metrics.loss_metric import Loss


class Model(nn.Module):
    """ A fully fledged model """
    def loss(self, x_data, y_true):
        """ Forward propagate network and return a value of loss function """
        y_pred = self(x_data)
        return y_pred, self.loss_value(x_data, y_true, y_pred)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        raise NotImplementedError

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss()]

