import itertools as it
import collections

import torch.nn as nn

from vel.api import BackboneModule, ModuleFactory, SizeHints
from vel.util.tensor_util import to_device
from .layer_base import LayerFactory, LayerFactoryContext


class LayerList(BackboneModule):
    """ Modification of nn.Sequential for the purpose of modular networks """
    def __init__(self, layers: collections.OrderedDict):
        super().__init__()

        self._layers = []

        for key, module in layers.items():
            self.add_module(key, module)
            self._layers.append(module)

        self._is_stateful = any(l.is_stateful for l in self._layers)

    def reset_weights(self):
        for l in self._layers:
            l.reset_weights()

    def size_hints(self) -> SizeHints:
        return self._layers[-1].size_hints()

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return self._is_stateful

    def zero_state(self, batch_size):
        """ Potential state for the model """
        zero_state = {}

        for l in self._layers:
            if l.is_stateful:
                layer_zero_state = l.zero_state(batch_size)
                if layer_zero_state is not None:
                    zero_state.update(layer_zero_state)

        return zero_state

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, item):
        return self._layers[item]

    def forward(self, direct, state: dict = None, context: dict = None):
        if not self._is_stateful:
            for layer in self._layers:
                direct = layer(direct, state=state, context=context)
            return direct
        else:
            data = direct
            output_state = {}

            for layer in self._layers:
                if layer.is_stateful:
                    data, new_state = layer(data, state=state, context=context)
                    output_state.update(new_state)
                else:
                    data = layer(data, state=state, context=context)

            return data, output_state


def instantiate_layers(layers: [LayerFactory], group: str, size_hint: SizeHints, extra_args: dict) -> nn.Module:
    """ Instantiate list of layer factories into PyTorch Module """
    module_dict = collections.OrderedDict()
    context_data = {}

    for idx, layer_factory in enumerate(layers):
        counter = idx + 1

        context = LayerFactoryContext(
            idx=counter,
            parent_group=group,
            parent_name=None,
            data=context_data
        )

        layer = layer_factory.instantiate(direct_input=size_hint, context=context, extra_args=extra_args)
        size_hint = layer.size_hints()

        module_dict[layer.name] = layer

    return LayerList(module_dict)


class ModularNetwork(BackboneModule):
    """ Network that is built from layers """

    def __init__(self, layers: LayerList):
        super().__init__()

        self.layers = layers
        assert not self.layers.is_stateful

    def reset_weights(self):
        """ Call proper initializers for the weights """
        self.layers.reset_weights()

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return False

    def size_hints(self) -> SizeHints:
        return self.layers.size_hints()

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return None

    def reset_state(self, state, dones):
        """ Reset the state after the episode has been terminated """
        raise NotImplementedError

    def forward(self, input_data, state=None, context: dict = None):
        return self.layers(input_data, state=None, context=context)

    def grouped_parameters(self):
        """ Return iterable of pairs (group, parameters) """
        return it.chain.from_iterable(l.grouped_parameters() for l in self.layers)


class StatefulModularNetwork(BackboneModule):
    """ Modular network handling the state between the episodes """

    def __init__(self, layers: LayerList):
        super().__init__()

        self.layers = layers

    def reset_weights(self):
        """ Call proper initializers for the weights """
        self.layers.reset_weights()

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return True

    def size_hints(self) -> SizeHints:
        return self.layers.size_hints()

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return self.layers.zero_state(batch_size)

    def reset_state(self, state, dones):
        """ Reset the state after the episode has been terminated """
        raise NotImplementedError

    def forward(self, input_data, state=None):
        data = input_data
        context = {}

        if state is None:
            # input_data.device here may break. Should be fixed at some point
            state = to_device(self.zero_state(input_data.size(0)), input_data.device)

        data, output_state = self.layers(data, state=state, context=context)

        return data, output_state

    def grouped_parameters(self):
        """ Return iterable of pairs (group, parameters) """
        return it.chain.from_iterable(l.grouped_parameters() for l in self.layers)


class ModularNetworkFactory(ModuleFactory):
    """ Factory class for the modular network """
    def __init__(self, layers: [LayerFactory], group=None):
        self.layers = layers

        if group is None:
            self.group = "default"
        else:
            self.group = group

    def instantiate(self, size_hint=None, **extra_args) -> BackboneModule:
        """ Create either stateful or not modular network instance """
        if size_hint is None:
            size_hint = SizeHints()

        layers = instantiate_layers(self.layers, self.group, size_hint=size_hint, extra_args=extra_args)
        is_stateful = any(l.is_stateful for l in layers)

        if is_stateful:
            return StatefulModularNetwork(layers)
        else:
            return ModularNetwork(layers)


def create(layers: [LayerFactory], group=None):
    """ Vel factory function """
    return ModularNetworkFactory(layers, group)
