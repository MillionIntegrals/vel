from vel.api import RnnLinearBackboneModel, ModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnn
from vel.modules.rnn_cell import RnnCell


class NatureCnnRnnBackbone(RnnLinearBackboneModel):
    """
    Long-Short-Term Memory rnn cell together with DeepMind-style 'Nature' cnn preprocessing
    """

    def __init__(self, input_width: int, input_height: int, input_channels: int, rnn_type='lstm',
                 cnn_output_dim: int=512, hidden_units: int=128):
        super().__init__()

        self.hidden_units = hidden_units

        self.nature_cnn = NatureCnn(input_width, input_height, input_channels, cnn_output_dim)
        self.rnn_cell = RnnCell(input_size=self.nature_cnn.output_dim, hidden_size=self.hidden_units, rnn_type=rnn_type)

    def reset_weights(self):
        """ Call proper initializers for the weights """
        self.nature_cnn.reset_weights()
        self.rnn_cell.reset_weights()

    @property
    def output_dim(self) -> int:
        return self.rnn_cell.output_dim

    @property
    def state_dim(self) -> int:
        """ Initial state of the network """
        return self.rnn_cell.state_dim

    def forward(self, input_image, state):
        cnn_output = self.nature_cnn(input_image)
        hidden_state, new_state = self.rnn_cell(cnn_output, state)

        return hidden_state, new_state


def create(input_width, input_height, input_channels=1, rnn_type='lstm', cnn_output_dim=512, hidden_units=128):
    """ Vel factory function """
    def instantiate(**_):
        return NatureCnnRnnBackbone(
            input_width=input_width, input_height=input_height, input_channels=input_channels,
            rnn_type=rnn_type, cnn_output_dim=cnn_output_dim, hidden_units=hidden_units
        )

    return ModelFactory.generic(instantiate)


# Add this to make nicer scripting interface
NatureCnnFactory = create
