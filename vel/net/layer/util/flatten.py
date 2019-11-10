import numpy as np

from vel.api import SizeHints, SizeHint
from vel.net.layer_base import LayerFactory, Layer, LayerFactoryContext, LayerInfo


class Flatten(Layer):
    """ Flatten single tensor to a unit shape """

    def __init__(self, info: LayerInfo, size_hint: SizeHint):
        super().__init__(info)

        self._size_hints = SizeHints(SizeHint(None, np.prod(size_hint[1:])))

    def forward(self, direct, state: dict = None, context: dict = None):
        return direct.view(direct.size(0), -1)

    def size_hints(self) -> SizeHints:
        return self._size_hints


class FlattenFactory(LayerFactory):
    """ Factory for Concat Layer """
    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "flatten"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        return Flatten(
            info=self.make_info(context),
            size_hint=direct_input.assert_single()
        )


def create(label=None, group=None):
    """ Vel factory function """
    return FlattenFactory().with_given_name(label).with_given_group(group)
