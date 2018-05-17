

class BaseMetric:
    """ Base class for all the metrics """

    def __init__(self, name):
        self.name = name

    def calculate(self, x_input, y_true, y_pred, **kwargs):
        """ Calculate value of a metric """
        raise NotImplementedError

    def reset(self):
        """ Reset value of a metric """
        raise NotImplementedError

    def value(self):
        """ Return current value for the metric """
        raise NotImplementedError
