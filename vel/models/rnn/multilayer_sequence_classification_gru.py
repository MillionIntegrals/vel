import typing

import torch
import torch.nn.functional as F
import torch.nn as nn

from vel.api.base import SupervisedModel, ModelFactory, LinearBackboneModel
from vel.metrics.accuracy import Accuracy
from vel.metrics.loss_metric import Loss


class MultilayerSequenceClassificationGRU(SupervisedModel):
    """ Multilayer GRU network for sequence modeling (n:1) """

    def __init__(self, input_block: LinearBackboneModel, output_dim: int,
                 rnn_layers: typing.List[int], rnn_dropout: float=0.0, bidirectional: bool=False,
                 linear_layers: typing.List[int]=None, linear_dropout: float=0.0):
        super().__init__()

        self.output_dim = output_dim

        self.rnn_layers_sizes = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.linear_layers_sizes = linear_layers
        self.linear_dropout = linear_dropout

        self.bidirectional = bidirectional
        self.input_block = input_block

        current_dim = self.input_block.output_dim

        self.rnn_layers = []
        self.rnn_dropout_layers = []

        bidirectional_multiplier = 1

        for idx, current_layer in enumerate(rnn_layers, 1):
            gru = nn.GRU(
                input_size=current_dim * bidirectional_multiplier,
                hidden_size=current_layer,
                bidirectional=bidirectional,
                batch_first=True,
            )

            self.add_module('gru{:02}'.format(idx), gru)
            self.rnn_layers.append(gru)

            if self.rnn_dropout > 0.0:
                dropout_layer = nn.Dropout(p=self.rnn_dropout)

                self.add_module('rnn_dropout{:02}'.format(idx), dropout_layer)
                self.rnn_dropout_layers.append(dropout_layer)

            current_dim = current_layer

            if self.bidirectional:
                bidirectional_multiplier = 2
            else:
                bidirectional_multiplier = 1

        self.linear_layers = []
        self.linear_dropout_layers = []

        for idx, current_layer in enumerate(linear_layers, 1):
            linear_layer = nn.Linear(current_dim * bidirectional_multiplier, current_layer)

            self.add_module('linear{:02}'.format(idx), linear_layer)
            self.linear_layers.append(linear_layer)

            if self.linear_dropout > 0.0:
                dropout_layer = nn.Dropout(p=self.linear_dropout)

                self.add_module('linear_dropout{:02}'.format(idx), dropout_layer)
                self.linear_dropout_layers.append(dropout_layer)

            bidirectional_multiplier = 1
            current_dim = current_layer

        if self.bidirectional:
            self.output_layer = nn.Linear(bidirectional_multiplier * current_dim, output_dim)
        else:
            self.output_layer = nn.Linear(current_dim, output_dim)

        self.output_activation = nn.LogSoftmax(dim=1)

    def reset_weights(self):
        self.input_block.reset_weights()

        for layer in self.linear_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, sequence):
        """ Forward propagate batch of sequences through the network, without accounting for the state """
        data = self.input_block(sequence)

        for idx in range(len(self.rnn_layers)):
            data, _ = self.rnn_layers[idx](data)

            if self.rnn_dropout_layers:
                data = self.rnn_dropout_layers[idx](data)

        # We are interested only in the last element of the sequence
        if self.bidirectional:
            last_hidden_size = self.rnn_layers_sizes[-1]
            data = torch.cat([data[:, -1, :last_hidden_size], data[:, 0, last_hidden_size:]], dim=1)
        else:
            data = data[:, -1]

        for idx in range(len(self.linear_layers_sizes)):
            data = F.relu(self.linear_layers[idx](data))

            if self.linear_dropout_layers:
                data = self.linear_dropout_layers[idx](data)

        data = self.output_layer(data)

        return self.output_activation(data)

    # def forward_state(self, sequence, state=None):
    #     """ Forward propagate a sequence through the network accounting for the state """
    #     if state is None:
    #         state = self.initial_state(sequence.size(0))
    #
    #     data = self.input_layer(sequence)
    #
    #     state_outputs = []
    #
    #     # for layer_length, layer in zip(self.hidden_layers, self.gru_layers):
    #     for idx in range(len(self.gru_layers)):
    #         layer_length = self.hidden_layers[idx]
    #
    #         # Partition hidden state, for each layer we have layer_length of h state and layer_length of c state
    #         current_state = state[:, :, :layer_length]
    #         state = state[:, :, layer_length:]
    #
    #         # Propagate through the GRU state
    #         data, new_h = self.gru_layers[idx](data, current_state)
    #
    #         if self.dropout_layers:
    #             data = self.dropout_layers[idx](data)
    #
    #         state_outputs.append(new_h)
    #
    #     output_data = self.output_activation(self.output_layer(data))
    #
    #     concatenated_hidden_output = torch.cat(state_outputs, dim=2)
    #
    #     return output_data, concatenated_hidden_output

    def get_layer_groups(self):
        return [
            self.input_block,
            self.rnn_layers,
            self.linear_layers,
            self.output_layer
        ]

    def initial_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, 1, sum(self.rnn_layers_sizes))

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self) -> list:
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]


def create(input_block: LinearBackboneModel, output_dim: int,
           rnn_layers: typing.List[int], rnn_dropout: float=0.0, bidirectional: bool=False,
           linear_layers: typing.List[int]=None, linear_dropout: float=0.0):
    """ Vel creation function """
    if linear_layers is None:
        linear_layers = []

    def instantiate(**_):
        return MultilayerSequenceClassificationGRU(
            input_block=input_block, output_dim=output_dim,
            rnn_layers=rnn_layers, rnn_dropout=rnn_dropout, bidirectional=bidirectional,
            linear_layers=linear_layers, linear_dropout=linear_dropout
        )

    return ModelFactory.generic(instantiate)
