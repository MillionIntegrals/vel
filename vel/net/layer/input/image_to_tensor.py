import typing

from vel.api import SizeHints, SizeHint
from vel.net.modular import LayerFactory, Layer
from vel.module.input.image_to_tensor import image_to_tensor


class ImageToTensorLayer(Layer):
    """
    Convert simple image to tensor.

    Flip channels to a [C, W, H] order and potentially convert 8-bit color values to floats
    """
    def __init__(self, name: str, size: tuple = None):
        super().__init__(name)

        if size is not None:
            assert len(size) == 3, "Images must have three dimensions"
            self.w, self.h, self.c = size
        else:
            self.w, self.h, self.c = (None, None, None)

    def forward(self, direct, state: dict = None, context: dict = None):
        return image_to_tensor(direct)

    def size_hints(self) -> SizeHints:
        return SizeHints(SizeHint(None, self.c, self.w, self.h))


class ImageToTensorLayerFactory(LayerFactory):
    def __init__(self, size: tuple = None):
        self.size = size

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "image_to_tensor"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict) -> Layer:
        """ Create a given layer object """
        # Potential improvement here is to use either direct input or size parameter
        return ImageToTensorLayer(name=name, size=self.size)


def create(size: tuple = None):
    """ Vel factory function """
    return ImageToTensorLayerFactory(size=size)
