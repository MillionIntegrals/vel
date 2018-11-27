import typing

import torch
import torch.nn.functional as F
import torch.nn as nn

from vel.api.base import SupervisedModel, ModelFactory, LinearBackboneModel


class MultilayerSequenceLSTM(SupervisedModel):
    """ Multilayer LSTM network for sequence modeling (n:n) """

    def __init__(self, input_block: LinearBackboneModel, hidden_layers: typing.List[int], output_dim: int,
                 dropout: float=0.0):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        self.input_block = input_block

        current_dim = self.input_block.output_dim

        self.lstm_layers = []
        self.dropout_layers = []

        for idx, current_layer in enumerate(hidden_layers, 1):
            lstm = nn.LSTM(
                input_size=current_dim,
                hidden_size=current_layer,
                batch_first=True,
                # dropout=dropout
            )

            self.add_module('lstm{:02}'.format(idx), lstm)
            self.lstm_layers.append(lstm)

            if dropout > 0.0:
                dropout_layer = nn.Dropout(p=dropout)

                self.add_module('dropout{:02}'.format(idx), dropout_layer)
                self.dropout_layers.append(dropout_layer)

            current_dim = current_layer

        self.output_layer = nn.Linear(current_dim, output_dim)
        self.output_activation = nn.LogSoftmax(dim=2)

    def reset_weights(self):
        self.input_block.reset_weights()

    def forward(self, sequence):
        """ Forward propagate batch of sequences through the network, without accounting for the state """
        data = self.input_block(sequence)

        for idx in range(len(self.lstm_layers)):
            data, _ = self.lstm_layers[idx](data)

            if self.dropout_layers:
                data = self.dropout_layers[idx](data)

        data = self.output_layer(data)

        return self.output_activation(data)

    def forward_state(self, sequence, state=None):
        """ Forward propagate a sequence through the network accounting for the state """
        if state is None:
            state = self.initial_state(sequence.size(0))

        data = self.input_layer(sequence)

        state_outputs = []

        for idx in range(len(self.lstm_layers)):
            layer_length = self.hidden_layers[idx]

            # Partition hidden state, for each layer we have layer_length of h state and layer_length of c state
            current_state = state[:, :, :layer_length * 2]
            state = state[:, :, 2 * layer_length:]

            # Split into h and c state
            current_h = current_state[:, :, :layer_length]
            current_c = current_state[:, :, layer_length:]

            # Propagate through the LSTM state
            data, (new_h, new_c) = self.lstm_layers[idx](data, (current_h, current_c))

            if self.dropout_layers:
                data = self.dropout_layers[idx](data)

            state_outputs.append(new_h)
            state_outputs.append(new_c)

        output_data = self.output_activation(self.output_layer(data))

        concatenated_hidden_output = torch.cat(state_outputs, dim=2)

        return output_data, concatenated_hidden_output

    def initial_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, 1, 2 * sum(self.hidden_layers))

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1).to(torch.long)
        return F.nll_loss(y_pred, y_true)


def create(input_block: LinearBackboneModel, hidden_layers: typing.List[int], output_dim: int, dropout=0.0):
    """ Vel creation function """
    def instantiate(**_):
        return MultilayerSequenceLSTM(
            input_block, hidden_layers, output_dim, dropout=dropout
        )

    return ModelFactory.generic(instantiate)
