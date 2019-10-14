import torch
import torch.nn as nn
import torch.nn.init as init


from vel.api import SizeHint, SizeHints
from vel.net.layer_base import Layer, LayerFactory, LayerFactoryContext, LayerInfo


class RnnCell(Layer):
    """ Generalization of RNN cell (Simple RNN, LSTM or GRU) """

    def __init__(self, info: LayerInfo, input_size: int, hidden_size: int, rnn_type: str, bias: bool = True,
                 nonlinearity: str = 'tanh'):
        super().__init__(info)

        assert rnn_type in {'rnn', 'lstm', 'gru'}, "Rnn type {} is not supported".format(rnn_type)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        if self.rnn_type == 'rnn':
            self.rnn_cell = nn.RNNCell(
                input_size=input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity
            )
        elif self.rnn_type == 'lstm':
            self.rnn_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
        elif self.rnn_type == 'gru':
            self.rnn_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return True

    def reset_weights(self):
        init.xavier_normal_(self.rnn_cell.weight_hh)
        init.xavier_normal_(self.rnn_cell.weight_ih)
        init.zeros_(self.rnn_cell.bias_ih)
        init.zeros_(self.rnn_cell.bias_hh)

    def size_hints(self) -> SizeHints:
        return SizeHints(SizeHint(None, self.hidden_size))

    @property
    def state_dim(self) -> int:
        """ Dimension of model state """
        if self.rnn_type == 'lstm':
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return {self.name: torch.zeros(batch_size, self.state_dim)}

    def forward(self, input_data, state: dict, context: dict = None):
        """ Forward propagation of a single layer """
        if self.rnn_type == 'lstm':
            state_tensor = state[self.name]
            hidden_state, cell_state = torch.split(state_tensor, self.hidden_size, 1)
            hidden_state, cell_state = self.rnn_cell(input_data, (hidden_state, cell_state))
            new_state = torch.cat([hidden_state, cell_state], dim=1)
            return hidden_state, {self.name: new_state}
        else:
            state_tensor = state[self.name]
            new_hidden_state = self.rnn_cell(input_data, state_tensor)
            return new_hidden_state, {self.name: new_hidden_state}


class RnnCellFactory(LayerFactory):
    """ Factory for the RnnCell layer """

    def __init__(self, hidden_size: int, rnn_type: str, bias: bool = True, nonlinearity: str = 'tanh'):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.bias = bias
        self.nonlinearity = nonlinearity

    @property
    def name_base(self) -> str:
        return "rnn_cell"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        input_size = direct_input.assert_single().last()

        return RnnCell(
            info=self.make_info(context),
            input_size=input_size,
            hidden_size=self.hidden_size,
            rnn_type=self.rnn_type,
            bias=self.bias,
            nonlinearity=self.nonlinearity
        )


def create(hidden_size: int, rnn_type: str, bias: bool = True, nonlinearity: str = 'tanh', label=None, group=None):
    """ Vel factory function """
    return RnnCellFactory(
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        bias=bias,
        nonlinearity=nonlinearity
    ).with_given_name(label).with_given_group(group)
