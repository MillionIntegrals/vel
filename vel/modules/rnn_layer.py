import torch
import torch.nn as nn
import torch.nn.init as init


from vel.api import RnnLinearBackboneModel


class RnnLayer(RnnLinearBackboneModel):
    """ Generalization of RNN layer (Simple RNN, LSTM or GRU) """

    def __init__(self, input_size, hidden_size, rnn_type, bias=True, bidirectional=False, nonlinearity='tanh'):
        super().__init__()

        assert rnn_type in {'rnn', 'lstm', 'gru'}, "RNN type {} is not supported".format(rnn_type)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        if self.rnn_type == 'rnn':
            self.rnn_cell = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity,
                bidirectional=bidirectional, batch_first=True
            )
        elif self.rnn_type == 'lstm':
            self.rnn_cell = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, bias=bias,
                bidirectional=bidirectional, batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn_cell = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, bias=bias,
                bidirectional=bidirectional, batch_first=True
            )

    def reset_weights(self):
        init.xavier_normal_(self.rnn_cell.weight_hh)
        init.xavier_normal_(self.rnn_cell.weight_ih)
        init.zeros_(self.rnn_cell.bias_ih)
        init.zeros_(self.rnn_cell.bias_hh)

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        if self.bidirectional:
            return 2.0 * self.hidden_size
        else:
            return self.hidden_size

    @property
    def state_dim(self) -> int:
        """ Dimension of model state """
        if self.rnn_type == 'lstm':
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def forward(self, input_data, state=None):
        if state is None:
            if self.bidirectional:
                state = self.zero_state(input_data.size(0)).unsqueeze(0).repeat(2, 1, 1).to(input_data.device)
            else:
                state = self.zero_state(input_data.size(0)).unsqueeze(0).to(input_data.device)

        if self.rnn_type == 'lstm':
            hidden_state, cell_state = torch.split(state, self.hidden_size, 2)
            hidden_state = hidden_state.contiguous()
            cell_state = cell_state.contiguous()
            output, (hidden_state, cell_state) = self.rnn_cell(input_data, (hidden_state, cell_state))
            new_state = torch.cat([hidden_state, cell_state], dim=2)
            return output, new_state
        else:
            return self.rnn_cell(input_data, state)



