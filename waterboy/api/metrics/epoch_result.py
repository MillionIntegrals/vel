
class EpochResultAccumulator:
    """ Result of a single epoch of training -- accumulator """
    def __init__(self, epoch_idx, metrics):
        self.epoch_idx = epoch_idx
        self.metrics = metrics

        self.frozen_results = {}

        self._reset_metrics()
        self.metrics_by_name = {m.name: m for m in self.metrics}

    def calculate(self, data_dict):
        """ Calculate metric values """
        for m in self.metrics:
            m.calculate(data_dict)

    def _reset_metrics(self):
        """ Internal API : reset state of metrics """
        for m in self.metrics:
            m.reset()

    def value(self):
        """ Return current value of the metrics """
        return {m.name: m.value() for m in self.metrics}

    def intermediate_value(self, metric):
        """ Return an intermediate (inter-epoch) value of a metric """
        if ':' in metric:
            metric_name = metric.split(':')[-1]
        else:
            metric_name = metric

        return self.metrics_by_name[metric_name].value()

    def freeze_results(self, name=None):
        new_results = self.value()

        if name is None:
            for key, value in new_results.items():
                self.frozen_results[key] = value
        else:
            for key, value in new_results.items():
                self.frozen_results[f'{name}:{key}'] = value

        self._reset_metrics()

    def result(self):
        """ Return the epoch result """
        final_result = {'epoch_idx': self.epoch_idx.global_epoch_idx}

        for key, value in self.frozen_results.items():
            final_result[key] = value

        return final_result
