import torch
import torch.nn as nn
import torch.nn.init as init

from vel.api.base import RnnLinearBackboneModel, ModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnn


class NatureCnnLstmBackbone(RnnLinearBackboneModel):
    """
    Long-Short-Term Memory rnn cell together with DeepMind-style 'Nature' cnn preprocessing
    """

    def __init__(self, input_width: int, input_height: int, input_channels: int, cnn_output_dim: int=512,
                 hidden_units: int=128):
        super().__init__()

        self.hidden_units = hidden_units

        self.nature_cnn = NatureCnn(input_width, input_height, input_channels, cnn_output_dim)
        self.lstm = nn.LSTMCell(input_size=512, hidden_size=self.hidden_units)

    def reset_weights(self):
        """ Call proper initializers for the weights """
        self.nature_cnn.reset_weights()

        init.orthogonal_(self.lstm.weight_ih, gain=1.0)
        init.orthogonal_(self.lstm.weight_hh, gain=1.0)
        init.zeros_(self.lstm.bias_ih)
        init.zeros_(self.lstm.bias_hh)

    @property
    def output_dim(self) -> int:
        return self.hidden_units

    @property
    def state_dim(self) -> int:
        """ Initial state of the network """
        return 2 * self.hidden_units

    def forward(self, input_image, state):
        cnn_output = self.nature_cnn(input_image)

        hidden_state, cell_state = torch.split(state, self.hidden_units, 1)
        hidden_state, cell_state = self.lstm(cnn_output, (hidden_state, cell_state))

        new_state = torch.cat([hidden_state, cell_state], dim=1)

        return hidden_state, new_state


def create(input_width, input_height, input_channels=1, cnn_output_dim=512, hidden_units=128):
    def instantiate(**_):
        return NatureCnnLstmBackbone(
            input_width=input_width, input_height=input_height, input_channels=input_channels,
            cnn_output_dim=cnn_output_dim, hidden_units=hidden_units
        )

    return ModelFactory.generic(instantiate)


# Add this to make nicer scripting interface
NatureCnnFactory = create
