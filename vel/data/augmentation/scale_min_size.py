"""
Code based on:
https://github.com/fastai/fastai/blob/master/fastai/transforms.py
"""
import vel.api as api
import vel.data.operation.image_op as op


class ScaleMinSize(api.ScopedTransformation):
    """ Scales the image so that the smallest axis is of 'size'. """
    def __init__(self, size, scope='x', tags=None):
        super().__init__(scope, tags)
        self.size = size

    def transform(self, x_data):
        return op.scale_min(x_data, self.size)


def create(size, scope='x', tags=None):
    """ Vel factory function """
    return ScaleMinSize(size, scope, tags)
