from vel.api import LinearBackboneModel, ModelFactory
from vel.module.rnn_cell import RnnCell


class RNN(LinearBackboneModel):
    """ Simple recurrent model backbone """

    def __init__(self, input_length: int, hidden_units: int, rnn_type: str = 'lstm'):
        super().__init__()

        self.input_length = input_length
        self.hidden_units = hidden_units

        self.rnn_cell = RnnCell(input_size=input_length, hidden_size=self.hidden_units, rnn_type=rnn_type)

    @property
    def output_dim(self) -> int:
        return self.rnn_cell.output_dim

    @property
    def state_dim(self) -> int:
        """ Initial state of the network """
        return self.rnn_cell.state_dim

    @property
    def is_stateful(self) -> bool:
        """ If the model has a state that needs to be fed between individual observations """
        return True

    def zero_state(self, batch_size):
        """ Potential state for the model """
        return self.rnn_cell.zero_state(batch_size)

    def forward(self, input_data, state):
        hidden_state, new_state = self.rnn_cell(input_data, state)
        return hidden_state, new_state


def create(input_length: int, hidden_units: int, rnn_type: str = 'lstm'):
    """ Vel factory function """
    def instantiate(**_):
        return RNN(
            input_length=input_length,
            hidden_units=hidden_units,
            rnn_type=rnn_type
        )
    return ModelFactory.generic(instantiate)
