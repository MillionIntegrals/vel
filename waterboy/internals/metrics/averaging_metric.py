import numpy as np

from .base_metric import BaseMetric


class AveragingMetric(BaseMetric):
    """ Base class for metrics that simply calculate the average over the epoch """
    def __init__(self, name):
        super().__init__(name)

        self.storage = []

    def calculate(self, x_input, y_true, y_pred, **kwargs):
        """ Calculate value of a metric """
        value = self._value_function(x_input, y_true, y_pred, **kwargs)
        self.storage.append(value)

    def _value_function(self, x_input, y_true, y_pred, **kwargs):
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        self.storage = []

    def value(self):
        """ Return current value for the metric """
        return np.mean(self.storage)
