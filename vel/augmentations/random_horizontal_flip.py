import random
import numpy as np

import vel.api.data as data


class RandomHorizontalFlip(data.Augmentation):
    """ Apply a horizontal flip randomly to input images """

    def __init__(self, p=0.5, mode='x', tags=None):
        super().__init__(mode, tags)
        self.p = p

    def __call__(self, img):
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


def create(p=0.5):
    return RandomHorizontalFlip(p)