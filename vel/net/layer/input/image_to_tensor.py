from vel.api import SizeHints, SizeHint
from vel.module.input.image_to_tensor import image_to_tensor
from vel.net.layer_base import LayerFactory, Layer, LayerFactoryContext, LayerInfo


class ImageToTensorLayer(Layer):
    """
    Convert simple image to tensor.

    Flip channels to a [C, W, H] order and potentially convert 8-bit color values to floats
    """
    def __init__(self, info: LayerInfo, shape: tuple = None):
        super().__init__(info)

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
        super().__init__()
        self.shape = shape

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "image_to_tensor"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        if self.shape is None:
            shape = direct_input.assert_single().shape()
        else:
            shape = self.shape

        return ImageToTensorLayer(
            info=self.make_info(context),
            shape=shape
        )


def create(shape: tuple = None, label=None, group=None):
    """ Vel factory function """
    return ImageToTensorLayerFactory(shape=shape).with_given_name(label).with_given_group(group)
