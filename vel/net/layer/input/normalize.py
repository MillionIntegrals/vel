import collections.abc as abc

from vel.api import SizeHints, SizeHint
from vel.module.input.normalize_observations import NormalizeObservations
from vel.net.layer_base import LayerFactory, Layer


class NormalizeLayer(Layer):
    """ Layer that normalizes the inputs """

    def __init__(self, name: str, shape):
        super().__init__(name)
        if not isinstance(shape, abc.Sequence):
            self.shape = (shape,)
        else:
            self.shape = shape

        self.normalize = NormalizeObservations(input_shape=shape)

    def forward(self, direct, state: dict = None, context: dict = None):
        return self.normalize(direct)

    def size_hints(self) -> SizeHints:
        return SizeHints(SizeHint(*([None] + list(self.shape))))


class NormalizeLayerFactory(LayerFactory):
    def __init__(self, shape=None):
        self.shape = shape

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "image_to_tensor"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict, extra_args: dict) -> Layer:
        """ Create a given layer object """
        # Potential improvement here is to use either direct input or size parameter
        if self.shape is None:
            shape = direct_input.assert_single().shape()
        else:
            shape = self.shape

        return NormalizeLayer(name=name, shape=shape)


def create(shape=None):
    """ Vel factory function """
    return NormalizeLayerFactory(shape=shape)
