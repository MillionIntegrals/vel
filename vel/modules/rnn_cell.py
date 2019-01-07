import torch
import torch.nn as nn
import torch.nn.init as init


from vel.api import RnnLinearBackboneModel


class RnnCell(RnnLinearBackboneModel):
    """ Generalization of RNN cell (Simple RNN, LSTM or GRU) """

    def __init__(self, input_size, hidden_size, rnn_type, bias=True, nonlinearity='tanh'):
        super().__init__()

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

    def reset_weights(self):
        init.xavier_normal_(self.rnn_cell.weight_hh)
        init.xavier_normal_(self.rnn_cell.weight_ih)
        init.zeros_(self.rnn_cell.bias_ih)
        init.zeros_(self.rnn_cell.bias_hh)

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self.hidden_size

    @property
    def state_dim(self) -> int:
        """ Dimension of model state """
        if self.rnn_type == 'lstm':
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def forward(self, input_data, state):
        if self.rnn_type == 'lstm':
            hidden_state, cell_state = torch.split(state, self.hidden_size, 1)
            hidden_state, cell_state = self.rnn_cell(input_data, (hidden_state, cell_state))
            new_state = torch.cat([hidden_state, cell_state], dim=1)
            return hidden_state, new_state
        else:
            new_hidden_state = self.rnn_cell(input_data, state)
            return new_hidden_state, new_hidden_state



