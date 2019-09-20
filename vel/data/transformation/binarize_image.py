import numpy as np

import vel.api as api


class BinarizeImage(api.ScopedTransformation):
    """ Convert [0,1] image into a binary {0, 1} representation """

    def transform(self, x_data):
        # Sample image from a Bernoulli distribution
        return np.random.binomial(1, x_data).astype(np.float32)


def create(scope='x', tags=None):
    """ Vel factory function """
    return BinarizeImage(scope, tags)
