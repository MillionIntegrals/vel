import typing
import collections
import torch.nn as nn

from vel.api import Network, BackboneNetwork, ModelFactory, SizeHints, SizeHint

from .layer_base import Layer, LayerFactory


def instantiate_layers(layers: [LayerFactory]) -> nn.Module:
    """ Instantiate list of layer factories into PyTorch Module """
    size_hint = SizeHints()  # Empty input at first
    module_dict = collections.OrderedDict()
    context = {}

    for idx, layer_factory in enumerate(layers):
        counter = idx + 1
        name = "{}_{:04d}".format(layer_factory.name_base, counter)

        layer = layer_factory.instantiate(name=name, direct_input=size_hint, context=context)
        size_hint = layer.size_hints()

        module_dict[name] = layer

    return nn.Sequential(module_dict)


class ModularNetwork(BackboneNetwork):
    """ Network that is built from layers """

    def __init__(self, layers: nn.Module):
        super().__init__()

        self.layers = layers
        assert not any(l.is_stateful for l in self.layers), "Does not support stateful layers"

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for l in self.layers:
            l.reset_weights()

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return False

    def size_hints(self) -> SizeHints:
        return self.layers[-1].size_hints()

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return None

    def reset_state(self, state, dones):
        """ Reset the state after the episode has been terminated """
        raise NotImplementedError

    def forward(self, input_data, state=None):
        return self.layers(input_data)


class StatefulModularNetwork(BackboneNetwork):
    """ Modular network handling the state between the episodes """

    def __init__(self, layers: nn.Module):
        super().__init__()

        self.layers = layers

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for l in self.layers:
            l.reset_weights()

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return True

    def size_hints(self) -> SizeHints:
        return self.layers[-1].size_hints()

    def zero_state(self, batch_size):
        """ Potential state for the model """
        raise NotImplementedError

    def reset_state(self, state, dones):
        """ Reset the state after the episode has been terminated """
        raise NotImplementedError

    def forward(self, input_data, state=None):
        raise NotImplementedError


class ModularNetworkFactory(ModelFactory):
    """ Factory class for the modular network """
    def __init__(self, layers: [LayerFactory]):
        self.layers = layers

    def instantiate(self, **extra_args) -> BackboneNetwork:
        """ Create either stateful or not modular network instance """
        layers = instantiate_layers(self.layers)
        is_stateful = any(l.is_stateful for l in layers)

        if is_stateful:
            return StatefulModularNetwork(layers)
        else:
            return ModularNetwork(layers)


def create(layers: [LayerFactory]):
    """ Vel factory function """
    return ModularNetworkFactory(layers)
