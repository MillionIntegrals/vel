import numpy as np

import torchvision.transforms.functional as F

import vel.api.data as data


class ToTensor(data.Augmentation):
    """ Convert image array to a tensor """
    def __init__(self, mode='x', tags=None):
        super().__init__(mode, tags)

    def __call__(self, datum):
        if len(datum.shape) == 2:
            # If the image has only one channel, it still needs to be specified
            datum = datum.reshape(datum.shape[0], datum.shape[1], 1)

        return F.to_tensor(datum)

    def denormalize(self, datum):
        return np.transpose(datum.numpy(), (1, 2, 0))


def create(mode='x', tags=None):
    """ Vel factory function """
    return ToTensor(mode, tags)
