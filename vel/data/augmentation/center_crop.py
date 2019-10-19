"""
Code based on:
https://github.com/fastai/fastai/blob/master/fastai/transforms.py
"""

import vel.api as api
import vel.data.operation.image_op as op


class CenterCrop(api.ScopedTransformation):
    """ A class that represents a Center Crop.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        sz: int
            size of the crop.
        tfm_y : TfmType
            type of y transformation.
    """
    def __init__(self, size, scope='x', tags=None):
        super().__init__(scope, tags)

        self.size = size

    def transform(self, x):
        return op.center_crop(x, self.size)


def create(size, scope='x', tags=None):
    return CenterCrop(size, scope, tags)
