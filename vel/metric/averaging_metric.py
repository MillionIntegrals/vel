import numpy as np

from .base_metric import BaseMetric


class AveragingMetric(BaseMetric):
    """ Base class for metrics that simply calculate the average over the epoch """
    def __init__(self, name):
        super().__init__(name)

        self.storage = []

    def calculate(self, batch_info):
        """ Calculate value of a metric """
        value = self._value_function(batch_info)
        self.storage.append(value)

    def _value_function(self, batch_info):
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        self.storage = []

    def value(self):
        """ Return current value for the metric """
        return float(np.mean(self.storage))


class AveragingNamedMetric(AveragingMetric):
    """ Super simple averaging metric that just takes a value from dictionary and averages it over samples """
    def __init__(self, name):
        super().__init__(name)

    def _value_function(self, batch_info):
        return batch_info[self.name]


class AveragingSupervisedMetric(BaseMetric):
    """ Base class for metrics that simply calculate the average over the epoch """
    def __init__(self, name):
        super().__init__(name)

        self.storage = []

    def calculate(self, batch_info):
        """ Calculate value of a metric """
        value = self._value_function(batch_info['data'], batch_info['target'], batch_info['output'])
        self.storage.append(value)

    def _value_function(self, x_input, y_true, y_pred):
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        self.storage = []

    def value(self):
        """ Return current value for the metric """
        return np.mean(self.storage)
