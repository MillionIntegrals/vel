import torch.nn as nn
import typing

from vel.api import SizeHints
from vel.net.layer_base import LayerFactory, Layer, LayerInfo, LayerFactoryContext


class ParallelLayer(Layer):
    """ Network that consists of parallel "towers" """

    def __init__(self, info: LayerInfo, layers: [Layer]):
        super().__init__(info)

        self.layers = nn.ModuleList(layers)
        self._size_hints = SizeHints(tuple(layer.size_hints().unwrap() for layer in self.layers))

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self._size_hints

    def forward(self, direct, state: dict = None, context: dict = None):
        """ Forward propagation of a single layer """
        results = [layer(x, state, context) for layer, x in zip(self.layers, direct)]
        return tuple(results)

    def grouped_parameters(self) -> typing.Iterable[(str, object)]:
        """ Return iterable of pairs (group, parameters) """
        raise NotImplementedError


class ParallelLayerFactory(LayerFactory):
    """ Factory for Parallel layer """

    def __init__(self, layers: [LayerFactory]):
        super().__init__()
        self.layers = layers

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "parallel"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        hints = direct_input.assert_tuple(len(self.layers))

        layers = []

        info = self.make_info(context)

        for idx, (size_hint, layer_factory) in enumerate(zip(hints, self.layers)):
            counter = idx + 1

            child_context = LayerFactoryContext(
                idx=counter,
                parent_group=info.group,
                parent_name=info.name,
                data=context.data
            )

            layer = layer_factory.instantiate(
                direct_input=SizeHints(size_hint),
                context=child_context,
                extra_args=extra_args
            )

            layers.append(layer)

        return ParallelLayer(
            info=info,
            layers=layers
        )


def create(layers: [LayerFactory], label=None, group=None):
    """ Vel factory function """
    return ParallelLayerFactory(layers=layers).with_given_name(label).with_given_group(group)
