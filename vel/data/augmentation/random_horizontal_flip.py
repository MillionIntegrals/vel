import random
import numpy as np

import vel.api as api


class RandomHorizontalFlip(api.ScopedTransformation):
    """ Apply a horizontal flip randomly to input images """

    def __init__(self, p=0.5, scope='x', tags=None):
        super().__init__(scope, tags)
        self.p = p

    def transform(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.fliplr(img).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def create(p=0.5, scope='x', tags=None):
    return RandomHorizontalFlip(p, scope=scope, tags=tags)
