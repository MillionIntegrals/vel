"""
Code based on:
https://github.com/fastai/fastai/blob/master/fastai/transforms.py
"""
import PIL.Image as Image

import vel.api.data as data


class ScaleMinSize(data.Augmentation):
    """ Scales the image so that the smallest axis is of 'size'. """
    def __init__(self, size, mode='x', tags=None):
        super().__init__(mode, tags)
        self.size = size

    def __call__(self, x_data):
        return data.scale_min(x_data, self.size, Image.BILINEAR)


def create(size, mode='x', tags=None):
    """ Vel factory function """
    return ScaleMinSize(size, mode, tags)
