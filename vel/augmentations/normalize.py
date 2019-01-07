import numpy as np

import vel.api.data as data


class Normalize(data.Augmentation):
    """ Normalize input mean and standard deviation """

    def __init__(self, mean, std, mode='x', tags=None):
        super().__init__(mode, tags)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, x_data):
        return (x_data - self.mean) / self.std

    def denormalize(self, x_data):
        """ Operation reverse to normalization """
        return x_data * self.std + self.mean


def create(mean, std, mode='x', tags=None):
    """ Vel factory function """
    return Normalize(mean=mean, std=std, mode=mode, tags=tags)

