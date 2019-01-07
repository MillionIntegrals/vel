"""
Code based on:
https://github.com/fastai/fastai/blob/master/fastai/transforms.py
"""
import cv2
import random

import vel.api.data as data


class RandomRotate(data.Augmentation):
    """ Rotate image randomly by an angle between (-deg, +deg) """
    def __init__(self, deg, p=0.75, mode='x', tags=None):
        super().__init__(mode, tags)
        self.deg = deg
        self.p = p

    def __call__(self, x_data):
        if random.random() < self.p:
            random_degree = random.uniform(-self.deg, self.deg)
            return data.rotate_img(x_data, random_degree, mode=cv2.BORDER_REFLECT)
        else:
            # No, don't do it
            return x_data


def create(deg, p=0.75, mode='x', tags=None):
    """ Vel factory function """
    return RandomRotate(deg, p, mode, tags)
