import torch.nn as nn

from vel.api import SizeHints
from vel.net.layer_base import LayerFactory, Layer


class ParallelLayer(Layer):
    """ Network that consists of parallel "towers" """

    def __init__(self, name: str, layers: [Layer]):
        super().__init__(name)

        self.layers = nn.ModuleList(layers)
        self._size_hints = SizeHints(tuple(layer.size_hints().unwrap() for layer in self.layers))

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self._size_hints

    def forward(self, direct, state: dict = None, context: dict = None):
        """ Forward propagation of a single layer """
        results = [layer(x, state, context) for layer, x in zip(self.layers, direct)]
        return tuple(results)


class ParallelLayerFactory(LayerFactory):
    """ Factory for Parallel layer """

    def __init__(self, layers: [LayerFactory]):
        self.layers = layers

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "parallel"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict) -> Layer:
        hints = direct_input.assert_tuple(len(self.layers))

        layers = []

        for idx, (size_hint, layer_factory) in enumerate(zip(hints, self.layers)):
            counter = idx + 1
            local_name = "{}_{:04d}".format(layer_factory.name_base, counter)
            global_name = f"{name}/{local_name}"

            layer = layer_factory.instantiate(name=global_name, direct_input=SizeHints(size_hint), context=context)
            layers.append(layer)

        return ParallelLayer(name, layers)


def create(layers: [LayerFactory]):
    """ Vel factory function """
    return ParallelLayerFactory(layers=layers)
