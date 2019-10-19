import numpy as np

from vel.api.transformation import ScopedTransformation


class ToArray(ScopedTransformation):
    """ Convert image to an array of floats """

    def transform(self, value):
        array = np.array(value)

        if array.dtype == np.uint8:
            return array.astype(np.float32) / 255.0
        else:
            return array


def create(mode='x', tags=None):
    return ToArray(mode, tags)
