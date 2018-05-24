from waterboy.api.metrics.averaging_metric import AveragingMetric


class Loss(AveragingMetric):
    """ Just a loss function """
    def __init__(self):
        super().__init__("loss")

    def _value_function(self, x_input, y_true, y_pred, **kwargs):
        """ Just forward a value of the loss"""
        return kwargs['loss'].item()
