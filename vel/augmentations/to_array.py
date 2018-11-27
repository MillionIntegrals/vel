import numpy as np

import vel.api.data as data


class ToArray(data.Augmentation):
    """ Convert imate to an array of floats """
    def __init__(self, mode='x', tags=None):
        super().__init__(mode, tags)

    def __call__(self, x_data):
        array = np.array(x_data)

        if array.dtype == np.uint8:
            return array.astype(np.float32) / 255.0
        else:
            return array


def create(mode='x', tags=None):
    return ToArray(mode, tags)
