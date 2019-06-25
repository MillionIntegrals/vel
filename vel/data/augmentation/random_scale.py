"""
Code based on:
https://github.com/fastai/fastai/blob/master/fastai/transforms.py
"""
import cv2
import collections.abc as abc
import random

import vel.api as api
import vel.data.operation.image_op as op


class RandomScale(api.ScopedTransformation):
    """ Scales the image so that the smallest axis is of 'size' times a random number between 1.0 and max_zoom. """
    def __init__(self, size, max_zoom, p=0.75, scope='x', tags=None):
        super().__init__(scope, tags)
        self.size = size
        self.max_zoom = max_zoom
        self.p = p

    def transform(self, x_data):
        if random.random() < self.p:
            # Yes, do it
            min_z = 1.
            max_z = self.max_zoom
            if isinstance(self.max_zoom, abc.Iterable):
                min_z, max_z = self.max_zoom

            mult = random.uniform(min_z, max_z)
        else:
            # No, don't do it
            mult = 1.0

        return op.scale_min(x_data, int(self.size * mult), cv2.INTER_AREA)


def create(size, max_zoom, p=0.75, scope='x', tags=None):
    """ Vel factory function """
    return RandomScale(size, max_zoom, p, scope, tags)
