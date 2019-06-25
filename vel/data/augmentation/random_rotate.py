"""
Code based on:
https://github.com/fastai/fastai/blob/master/fastai/transforms.py
"""
import cv2
import random

import vel.api as api
import vel.data.operation.image_op as op


class RandomRotate(api.ScopedTransformation):
    """ Rotate image randomly by an angle between (-deg, +deg) """
    def __init__(self, deg, p=0.75, scope='x', tags=None):
        super().__init__(scope, tags)
        self.deg = deg
        self.p = p

    def transform(self, x_data):
        if random.random() < self.p:
            random_degree = random.uniform(-self.deg, self.deg)
            return op.rotate_img(x_data, random_degree, mode=cv2.BORDER_REFLECT)
        else:
            # No, don't do it
            return x_data


def create(deg, p=0.75, scope='x', tags=None):
    """ Vel factory function """
    return RandomRotate(deg, p, scope, tags)
