from vel.api.metrics.averaging_metric import AveragingSupervisedMetric


class Accuracy(AveragingSupervisedMetric):
    """ Classification accuracy """
    def __init__(self):
        super().__init__("accuracy")

    def _value_function(self, x_input, y_true, y_pred):
        """ Return classification accuracy of input """
        if len(y_true.shape) == 1:
            return y_pred.argmax(1).eq(y_true).double().mean().item()
        else:
            raise NotImplementedError


def create():
    """ Vel factory function """
    return Accuracy()
