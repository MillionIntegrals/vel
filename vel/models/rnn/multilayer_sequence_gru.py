import typing

import torch
import torch.nn.functional as F
import torch.nn as nn

from vel.api.base import SupervisedModel, ModelFactory, LinearBackboneModel


class MultilayerSequenceGRU(SupervisedModel):
    """ Multilayer GRU network for sequence modeling (n:n) """

    def __init__(self, input_block: LinearBackboneModel, hidden_layers: typing.List[int], output_dim: int,
                 dropout: float=0.0):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        self.input_block = input_block

        current_dim = self.input_block.output_dim

        self.gru_layers = []
        self.dropout_layers = []

        for idx, current_layer in enumerate(hidden_layers, 1):
            gru = nn.GRU(
                input_size=current_dim,
                hidden_size=current_layer,
                batch_first=True,
            )

            self.add_module('gru{:02}'.format(idx), gru)
            self.gru_layers.append(gru)

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

        for idx in range(len(self.gru_layers)):
            data, _ = self.gru_layers[idx](data)

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

        # for layer_length, layer in zip(self.hidden_layers, self.gru_layers):
        for idx in range(len(self.gru_layers)):
            layer_length = self.hidden_layers[idx]

            # Partition hidden state, for each layer we have layer_length of h state and layer_length of c state
            current_state = state[:, :, :layer_length]
            state = state[:, :, layer_length:]

            # Propagate through the GRU state
            data, new_h = self.gru_layers[idx](data, current_state)

            if self.dropout_layers:
                data = self.dropout_layers[idx](data)

            state_outputs.append(new_h)

        output_data = self.output_activation(self.output_layer(data))

        concatenated_hidden_output = torch.cat(state_outputs, dim=2)

        return output_data, concatenated_hidden_output

    def initial_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, 1, sum(self.hidden_layers))

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1).to(torch.long)
        return F.nll_loss(y_pred, y_true)


def create(input_block: LinearBackboneModel, hidden_layers: typing.List[int], output_dim: int, dropout=0.0):
    """ Vel creation function """
    def instantiate(**_):
        return MultilayerSequenceGRU(
            input_block, hidden_layers, output_dim, dropout=dropout
        )

    return ModelFactory.generic(instantiate)
