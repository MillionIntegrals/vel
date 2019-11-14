import collections

from vel.api import BackboneModule, SizeHints


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
