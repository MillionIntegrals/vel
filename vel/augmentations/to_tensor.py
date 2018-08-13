import numpy as np

import torchvision.transforms.functional as F

import vel.api.data as data


class ToTensor(data.Augmentation):
    def __init__(self, mode='x', tags=None):
        super().__init__(mode, tags)

    def __call__(self, datum):
        return F.to_tensor(datum)

    def denormalize(self, datum):
        return np.transpose(datum.numpy(), (1, 2, 0))


def create(mode='x', tags=None):
    return ToTensor(mode, tags)
