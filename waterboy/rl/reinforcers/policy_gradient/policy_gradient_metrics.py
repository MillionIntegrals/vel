import collections
import time

import numpy as np
import torch

from waterboy.api.metrics import BaseMetric
from waterboy.api.metrics.averaging_metric import AveragingMetric


class FPSMetric(AveragingMetric):
    """ Metric calculating FPS values """
    def __init__(self):
        super().__init__('fps')

        self.start_time = time.time()
        self.frames = 0

    def _value_function(self, data_dict):
        self.frames += data_dict['frames'].item()

        nseconds = time.time()-self.start_time
        fps = int(self.frames/nseconds)
        return fps


class EpisodeRewardMetric(BaseMetric):
    def __init__(self):
        super().__init__("episode_reward")
        self.buffer = collections.deque(maxlen=100)

    def calculate(self, data_dict):
        """ Calculate value of a metric based on supplied data """
        self.buffer.extend(data_dict['episode_infos'])

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


class EpisodeLengthMetric(BaseMetric):
    def __init__(self):
        super().__init__("episode_length")
        self.buffer = collections.deque(maxlen=100)

    def calculate(self, data_dict):
        """ Calculate value of a metric based on supplied data """
        self.buffer.extend(data_dict['episode_infos'])

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

    def _value_function(self, data_dict):
        values = data_dict['values']
        rewards = data_dict['rewards']

        explained_variance = 1 - torch.var(rewards - values) / torch.var(rewards)
        return explained_variance.item()
