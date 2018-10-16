import collections

import numpy as np
import torch

from vel.api import BatchInfo
from vel.api.metrics import BaseMetric, AveragingMetric, ValueMetric


class FramesMetric(ValueMetric):
    """ Count the frames """
    def __init__(self, name="frames"):
        super().__init__(name)

    def _value_function(self, batch_info: BatchInfo):
        return batch_info.training_info['frames']


class FPSMetric(ValueMetric):
    """ Metric calculating FPS values """
    def __init__(self, name='fps'):
        super().__init__(name)

    def _value_function(self, batch_info):
        frames = batch_info.training_info['frames']
        seconds = batch_info.training_info['time']

        fps = int(frames/seconds)

        return fps


class EpisodeRewardMetric(BaseMetric):
    def __init__(self, name):
        super().__init__(name)
        self.buffer = collections.deque(maxlen=100)

    def calculate(self, batch_info):
        """ Calculate value of a metric based on supplied data """
        self.buffer.extend(batch_info['episode_infos'])

    def reset(self):
        """ Reset value of a metric """
        # Because it's a queue no need for reset..
        pass

    def value(self):
        """ Return current value for the metric """
        if self.buffer:
            return np.mean([ep['r'] for ep in self.buffer])
        else:
            return 0.0


class EpisodeRewardMetricQuantile(BaseMetric):
    def __init__(self, name, quantile, buf_size=100):
        super().__init__(name)
        self.buffer = collections.deque(maxlen=buf_size)
        self.quantile = quantile

    def calculate(self, batch_info):
        """ Calculate value of a metric based on supplied data """
        self.buffer.extend(ep['r'] for ep in batch_info['episode_infos'])

    def reset(self):
        """ Reset value of a metric """
        # Because it's a queue no need for reset..
        pass

    def value(self):
        """ Return current value for the metric """
        if self.buffer:
            return np.quantile(self.buffer, self.quantile)
        else:
            return 0.0


class EpisodeLengthMetric(BaseMetric):
    def __init__(self, name):
        super().__init__(name)
        self.buffer = collections.deque(maxlen=100)

    def calculate(self, batch_info):
        """ Calculate value of a metric based on supplied data """
        self.buffer.extend(batch_info['episode_infos'])

    def reset(self):
        """ Reset value of a metric """
        # Because it's a queue no need for reset..
        pass

    def value(self):
        """ Return current value for the metric """
        if self.buffer:
            return np.mean([ep['l'] for ep in self.buffer])
        else:
            return 0


class ExplainedVariance(AveragingMetric):
    """ How much value do rewards explain """
    def __init__(self):
        super().__init__("explained_variance")

    def _value_function(self, batch_info):
        values = batch_info['values']
        rewards = batch_info['rewards']

        explained_variance = 1 - torch.var(rewards - values) / torch.var(rewards)
        return explained_variance.item()
