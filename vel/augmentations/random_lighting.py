import random

import vel.api.data as data


class RandomLighting(data.Augmentation):
    """ Apply a horizontal flip randomly to input images """

    def __init__(self, b, c, mode='x', tags=None):
        super().__init__(mode, tags)
        self.b, self.c = b, c

    def __call__(self, img):
        """ Adjust lighting """
        rand_b = random.uniform(-self.b, self.b)
        rand_c = random.uniform(-self.c, self.c)
        rand_c = -1/(rand_c-1) if rand_c<0 else rand_c+1
        return data.lighting(img, rand_b, rand_c)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def create(b, c, mode='x', tags=None):
    return RandomLighting(b, c, mode, tags)
