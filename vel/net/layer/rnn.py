import torch
import torch.nn as nn
import torch.nn.init as init

from vel.api import SizeHints
from vel.net.layer_base import Layer, LayerFactory


class RnnLayer(Layer):
    """ Single Recurrent Layer """
    def __init__(self, name: str, input_size: SizeHints, hidden_size: int, rnn_type: str,
                 bias: bool = True, bidirectional: bool = False, nonlinearity: str = 'tanh'):
        super().__init__(name)

        self.input_size = input_size
        self.input_length = input_size.assert_single().last()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.bias = bias
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity

        if self.rnn_type == 'rnn':
            self.rnn_cell = nn.RNN(
                input_size=self.input_length, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity,
                bidirectional=bidirectional, batch_first=True
            )
        elif self.rnn_type == 'lstm':
            self.rnn_cell = nn.LSTM(
                input_size=self.input_length, hidden_size=hidden_size, bias=bias,
                bidirectional=bidirectional, batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn_cell = nn.GRU(
                input_size=self.input_length, hidden_size=hidden_size, bias=bias,
                bidirectional=bidirectional, batch_first=True
            )

        self.output_size = input_size.assert_single().drop_last().append(self.hidden_size)

    def reset_weights(self):
        """ Call proper initializers for the weights """
        init.xavier_normal_(self.rnn_cell.weight_hh)
        init.xavier_normal_(self.rnn_cell.weight_ih)
        init.zeros_(self.rnn_cell.bias_ih)
        init.zeros_(self.rnn_cell.bias_hh)

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return True

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
            state_tensor = state[self.name].unsqueeze(0)
            hidden_state, cell_state = torch.split(state_tensor, self.hidden_size, dim=2)
            output, (hidden_state, cell_state) = self.rnn_cell(
                input_data, (hidden_state.contiguous(), cell_state.contiguous())
            )
            new_state = torch.cat([hidden_state, cell_state], dim=2)
            return output, {self.name: new_state[0]}
        else:
            state_tensor = state[self.name].unsqueeze(0)
            output, new_state = self.rnn_cell(input_data, state_tensor)
            return output, {self.name: new_state[0]}

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return SizeHints(self.output_size)


class RnnLayerFactory(LayerFactory):
    """ Factory class for the RnnLayer """

    def __init__(self, hidden_size: int, rnn_type: str, bias: bool = True, bidirectional: bool = False,
                 nonlinearity: str = 'tanh'):
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.bias = bias
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "rnn"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict, extra_args: dict) -> Layer:
        """ Create instance of 'RnnLayer' """
        return RnnLayer(
            name=name,
            input_size=direct_input,
            hidden_size=self.hidden_size,
            rnn_type=self.rnn_type,
            bias=self.bias,
            bidirectional=self.bidirectional,
            nonlinearity=self.nonlinearity
        )


def create(hidden_size: int, rnn_type: str, bias: bool = True, bidirectional: bool = False,
           nonlinearity: str = 'tanh'):
    """ Vel factory function """
    return RnnLayerFactory(
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        bias=bias,
        bidirectional=bidirectional,
        nonlinearity=nonlinearity
    )
