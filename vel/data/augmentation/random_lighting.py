import random

import vel.api as api
import vel.data.operation.image_op as op


class RandomLighting(api.ScopedTransformation):
    """ Apply a horizontal flip randomly to input images """

    def __init__(self, b, c, mode='x', tags=None):
        super().__init__(mode, tags)
        self.b, self.c = b, c

    def transform(self, img):
        """ Adjust lighting """
        rand_b = random.uniform(-self.b, self.b)
        rand_c = random.uniform(-self.c, self.c)
        rand_c = -1/(rand_c-1) if rand_c < 0 else rand_c+1
        return op.lighting(img, rand_b, rand_c)

    def __repr__(self):
        return self.__class__.__name__ + '(b={}, c={})'.format(self.b, self.c)


def create(b, c, scope='x', tags=None):
    return RandomLighting(b, c, scope, tags)
