from vel.api import SizeHints, SizeHint
from vel.module.input.normalize_expanding import NormalizeExpanding
from vel.net.layer_base import LayerFactory, Layer, LayerFactoryContext, LayerInfo


class NormalizeLayer(Layer):
    """ Layer that normalizes the inputs """

    def __init__(self, info: LayerInfo, input_shape: SizeHints):
        super().__init__(info)

        self.input_shape = input_shape

        self.normalize = NormalizeExpanding(
            input_shape=self.input_shape.assert_single()[1:]  # Remove batch axis
        )

    def forward(self, direct, state: dict = None, context: dict = None):
        return self.normalize(direct)

    def size_hints(self) -> SizeHints:
        return self.input_shape


class NormalizeLayerFactory(LayerFactory):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "image_to_tensor"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        if self.shape is None:
            input_shape = direct_input
        else:
            input_shape = SizeHints(SizeHint(*([None] + list(self.shape))))

        return NormalizeLayer(
            info=self.make_info(context),
            input_shape=input_shape
        )


def create(shape=None):
    """ Vel factory function """
    return NormalizeLayerFactory(shape=shape)
