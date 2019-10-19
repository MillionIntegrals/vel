import collections
import typing

from vel.api import BackboneModule, SizeHints
from vel.exception import VelException
from vel.util.tensor_util import to_device


class GenericModularSequential(BackboneModule):
    """ Modification of nn.Sequential for the purpose of modular networks """

    def __init__(self, layers: typing.Union[collections.OrderedDict, collections.Sequence]):
        super().__init__()
        self._layers = []

        if isinstance(layers, collections.OrderedDict):
            for key, module in layers.items():
                self.add_module(key, module)
                self._layers.append(module)
        elif isinstance(layers, collections.Sequence):
            for idx, module in enumerate(layers):
                key = str(idx)
                self.add_module(key, module)
                self._layers.append(module)
        else:
            raise VelException("Incorrectly specified layers, must be a sequence or an ordered dict")

        self._is_stateful = any(l.is_stateful() for l in self._layers)

    def size_hints(self) -> SizeHints:
        return self._layers[-1].size_hints()

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return self._is_stateful

    def zero_state(self, batch_size):
        """ Potential state for the model """
        zero_state = {}

        for l in self.layers:
            layer_zero_state = l.zero_state(batch_size)
            if layer_zero_state is not None:
                zero_state.update(layer_zero_state)

        return zero_state

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, item):
        return self._layers[item]

    def forward(self, direct, state: dict = None, context: dict = None):
        if not self.is_stateful:
            for layer in self._layers:
                direct = layer(direct, state=state, context=context)
            return direct
        else:
            output_state = {}

            if state is None:
                # direct.device here may break. Should be fixed at some point
                state = to_device(self.zero_state(direct.size(0)), direct.device)

            data = direct

            for layer in self.layers:
                if layer.is_stateful:
                    data, new_state = layer(data, state=state, context=context)
                    output_state.update(new_state)
                else:
                    data = layer(data, state=state, context=context)

            return data, output_state
