from vel.metric.base.averaging_metric import AveragingMetric


class SamplesPerSec(AveragingMetric):
    """ Just a loss function """
    def __init__(self, scope="train"):
        super().__init__("samples_per_sec", scope=scope)

    def _value_function(self, batch_info):
        """ Just forward a value of the loss"""
        return batch_info['samples'] / batch_info['time']
