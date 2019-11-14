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
        self._is_stateful = any(l.is_stateful for l in self.layers)

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return self._is_stateful

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self._size_hints

    def forward(self, direct, state: dict = None, context: dict = None):
        """ Forward propagation of a single layer """
        if self._is_stateful:
            results = []
            output_state = {}

            for layer, layer_input in zip(self.layers, direct):
                data, new_state = layer(layer_input, state=state, context=context)
                results.append(data)
                output_state.update(new_state)

            return tuple(results), output_state
        else:
            results = [layer(x, state, context) for layer, x in zip(self.layers, direct)]
            return tuple(results)

    def zero_state(self, batch_size):
        """ Potential state for the model """
        zero_state = {}

        for l in self.layers:
            if l.is_stateful:
                layer_zero_state = l.zero_state(batch_size)
                if layer_zero_state is not None:
                    zero_state.update(layer_zero_state)

        return zero_state

    def grouped_parameters(self) -> typing.Iterable[typing.Tuple[str, object]]:
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
