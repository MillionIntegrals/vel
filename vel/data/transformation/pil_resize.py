import PIL.Image as Image

import vel.api as api


class PilResize(api.ScopedTransformation):
    """ Scale the PIL image """
    def __init__(self, shape, scope='x', tags=None):
        super().__init__(scope, tags)
        self.shape = shape

    def transform(self, x_data):
        return x_data.resize(self.shape, Image.LANCZOS)


def create(shape, scope='x', tags=None):
    """ Vel factory function """
    return PilResize(shape, scope, tags)
