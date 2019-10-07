from vel.api import SizeHints, SizeHint
from vel.module.input.image_to_tensor import image_to_tensor
from vel.net.layer_base import LayerFactory, Layer


class ImageToTensorLayer(Layer):
    """
    Convert simple image to tensor.

    Flip channels to a [C, W, H] order and potentially convert 8-bit color values to floats
    """
    def __init__(self, name: str, shape: tuple = None):
        super().__init__(name)

        if shape is not None:
            assert len(shape) == 3, "Images must have three dimensions"
            self.w, self.h, self.c = shape
        else:
            self.w, self.h, self.c = (None, None, None)

    def forward(self, direct, state: dict = None, context: dict = None):
        return image_to_tensor(direct)

    def size_hints(self) -> SizeHints:
        return SizeHints(SizeHint(None, self.c, self.w, self.h))


class ImageToTensorLayerFactory(LayerFactory):
    def __init__(self, shape: tuple = None):
        self.shape = shape

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "image_to_tensor"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict, extra_args: dict) -> Layer:
        """ Create a given layer object """
        if self.shape is None:
            shape = direct_input.assert_single().shape()
        else:
            shape = self.shape

        return ImageToTensorLayer(name=name, shape=shape)


def create(shape: tuple = None):
    """ Vel factory function """
    return ImageToTensorLayerFactory(shape=shape)
