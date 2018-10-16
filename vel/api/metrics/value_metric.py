from .base_metric import BaseMetric


class ValueMetric(BaseMetric):
    """ Base class for metrics that don't have state and just calculate a simple value """

    def __init__(self, name):
        super().__init__(name)

        self._metric_value = None

    def calculate(self, batch_info):
        """ Calculate value of a metric based on supplied data """
        self._metric_value = self._value_function(batch_info)

    def reset(self):
        """ Reset value of a metric """
        pass

    def value(self):
        """ Return current value for the metric """
        return self._metric_value

    def _value_function(self, batch_info):
        raise NotImplementedError

