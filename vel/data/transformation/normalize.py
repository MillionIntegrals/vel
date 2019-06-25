import numpy as np

import vel.api as api


class Normalize(api.ScopedTransformation):
    """ Normalize input mean and standard deviation """

    def __init__(self, mean=None, std=None, scope='x', tags=None):
        super().__init__(scope, tags)

        self.mean = mean
        self.std = std

        if self.mean is not None:
            self.mean = np.asarray(self.mean)

        if self.std is not None:
            self.std = np.asarray(self.std)

    def initialize(self, source):
        """ Initialize transformation from source """
        if self.mean is None:
            self.mean = source.metadata['train_mean']

        if self.std is None:
            self.std = source.metadata['train_std']

    def transform(self, value):
        return (value - self.mean) / self.std

    def denormalization_transform(self, value):
        """ Operation reverse to normalization """
        return value * self.std + self.mean


def create(mean=None, std=None, mode='x', tags=None):
    """ Vel factory function """
    return Normalize(mean=mean, std=std, scope=mode, tags=tags)
