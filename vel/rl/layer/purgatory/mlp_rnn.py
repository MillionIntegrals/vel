import typing

from vel.api import LinearBackboneModel, ModelFactory
from vel.rl.backbone.mlp import MLP
from vel.rl.backbone.rnn import RNN


class MlpRnn(LinearBackboneModel):
    """ MLP followed by an RNN - another simple policy backbone """

    def __init__(self, input_length: int, mlp_layers: typing.List[int], rnn_units: int, rnn_type: str = 'lstm',
                 mlp_activation: str = 'tanh', mlp_normalization: typing.Optional[str] = None):
        super().__init__()

        self.mlp = MLP(
            input_length=input_length, hidden_layers=mlp_layers, activation=mlp_activation,
            normalization=mlp_normalization
        )

        self.rnn = RNN(input_length=self.mlp.output_dim, hidden_units=rnn_units, rnn_type=rnn_type)

    @property
    def output_dim(self) -> int:
        return self.rnn.output_dim

    @property
    def state_dim(self) -> int:
        """ Initial state of the network """
        return self.rnn.state_dim

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return True

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return self.rnn.zero_state(batch_size)

    def forward(self, input_data, state):
        mlp_output = self.mlp(input_data)
        hidden_state, new_state = self.rnn(mlp_output, state)
        return hidden_state, new_state


def create(input_length: int, mlp_layers: typing.List[int], rnn_units: int, rnn_type: str = 'lstm',
           mlp_activation: str = 'tanh', mlp_normalization: typing.Optional[str] = None):
    """ Vel factory function """
    def instantiate(**_):
        return MlpRnn(
            input_length=input_length,
            mlp_layers=mlp_layers,
            rnn_units=rnn_units,
            rnn_type=rnn_type,
            mlp_activation=mlp_activation,
            mlp_normalization=mlp_normalization
        )

    return ModelFactory.generic(instantiate)
