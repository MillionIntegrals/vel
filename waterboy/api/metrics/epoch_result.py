
class EpochResultAccumulator:
    """ Result of a single epoch of training -- accumulator """
    def __init__(self, epoch_idx, metrics):
        self.epoch_idx = epoch_idx
        self.metrics = metrics

        self.train_results = {}
        self.validation_results = {}

        self._reset_metrics()
        self.metrics_by_name = {m.name: m for m in self.metrics}

    def calculate(self, x_data, y_true, y_pred, **kwargs):
        """ Calculate metric values """
        for m in self.metrics:
            m.calculate(x_data, y_true, y_pred, **kwargs)

    def _reset_metrics(self):
        """ Internal API : reset state of metrics """
        for m in self.metrics:
            m.reset()

    def value(self):
        """ Return current value of the metrics """
        return {m.name: m.value() for m in self.metrics}

    # def value_string(self, precision=6):
    #     """ Return a string describing current values of all metrics """
    #     return " ".join([("{}: {:." + str(precision) + "f}").format(m.name, m.value()) for m in self.metrics])

    def intermediate_value(self, metric):
        """ Return an intermediate (inter-epoch) value of a metric """
        if ':' in metric:
            metric_name = metric.split(':')[-1]
        else:
            metric_name = metric

        return self.metrics_by_name[metric_name].value()

    def freeze_train_results(self):
        self.train_results = self.value()
        self._reset_metrics()

    def freeze_validation_results(self):
        self.validation_results = self.value()
        self._reset_metrics()

    def result(self):
        """ Return the epoch result """
        final_result = {'epoch_idx': self.epoch_idx}

        for key, value in self.train_results.items():
            final_result["train:{}".format(key)] = value

        for key, value in self.validation_results.items():
            final_result["val:{}".format(key)] = value

        return final_result


class EpochResult:
    """ Result of a single epoch of training """
    def __init__(self, data):
        self.data = data
