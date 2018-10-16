from vel.api.metrics.averaging_metric import AveragingMetric


class Loss(AveragingMetric):
    """ Just a loss function """
    def __init__(self):
        super().__init__("loss")

    def _value_function(self, batch_info):
        """ Just forward a value of the loss"""
        return batch_info['loss'].item()
