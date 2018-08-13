"""
Code based on:
https://github.com/fastai/fastai/blob/master/fastai/transforms.py
"""

import vel.api.data as data


class CenterCrop(data.Augmentation):
    """ A class that represents a Center Crop.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        sz: int
            size of the crop.
        tfm_y : TfmType
            type of y transformation.
    """
    def __init__(self, size, mode='x', tags=None):
        super().__init__(mode, tags)

        self.size = size

    def __call__(self, x):
        return data.center_crop(x, self.size)


def create(size, mode='x', tags=None):
    return CenterCrop(size, mode, tags)
