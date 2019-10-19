import torch
import torch.nn as nn
import torch.nn.init as init

from vel.api import SizeHints
from vel.net.layer_base import Layer, LayerFactory, LayerFactoryContext, LayerInfo


class RnnLayer(Layer):
    """ Single Recurrent Layer """
    def __init__(self, info: LayerInfo, input_shape: SizeHints, hidden_size: int, rnn_type: str,
                 bias: bool = True, bidirectional: bool = False, nonlinearity: str = 'tanh'):
        super().__init__(info)

        self.input_shape = input_shape
        self.input_length = input_shape.assert_single().last()
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

        if self.bidirectional:
            self.output_shape = SizeHints(input_shape.assert_single().drop_last().append(2 * self.hidden_size))
        else:
            self.output_shape = SizeHints(input_shape.assert_single().drop_last().append(self.hidden_size))

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
        if self.bidirectional:
            return {self.global_name: torch.zeros(2, batch_size, self.state_dim)}
        else:
            return {self.global_name: torch.zeros(1, batch_size, self.state_dim)}

    def forward(self, input_data, state: dict, context: dict = None):
        """ Forward propagation of a single layer """

        if self.rnn_type == 'lstm':
            state_tensor = state[self.name]
            hidden_state, cell_state = torch.split(state_tensor, self.hidden_size, dim=2)
            output, (hidden_state, cell_state) = self.rnn_cell(
                input_data, (hidden_state.contiguous(), cell_state.contiguous())
            )
            new_state = torch.cat([hidden_state, cell_state], dim=2)
            return output, {self.name: new_state}
        else:
            state_tensor = state[self.name]
            output, new_state = self.rnn_cell(input_data, state_tensor)
            return output, {self.name: new_state}

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.output_shape


class RnnLayerFactory(LayerFactory):
    """ Factory class for the RnnLayer """

    def __init__(self, hidden_size: int, rnn_type: str, bias: bool = True, bidirectional: bool = False,
                 nonlinearity: str = 'tanh'):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.bias = bias
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "rnn"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create instance of 'RnnLayer' """
        return RnnLayer(
            info=self.make_info(context),
            input_shape=direct_input,
            hidden_size=self.hidden_size,
            rnn_type=self.rnn_type,
            bias=self.bias,
            bidirectional=self.bidirectional,
            nonlinearity=self.nonlinearity
        )


def create(hidden_size: int, rnn_type: str, bias: bool = True, bidirectional: bool = False,
           nonlinearity: str = 'tanh', label=None, group=None):
    """ Vel factory function """
    return RnnLayerFactory(
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        bias=bias,
        bidirectional=bidirectional,
        nonlinearity=nonlinearity
    ).with_given_name(label).with_given_group(group)
