from vel.api import LinearBackboneModel, ModelFactory


class LstmBackbone(LinearBackboneModel):
    """
    Simple 'LSTM' model backbone
    """

    def __init__(self, input_size, hidden_units):
        super().__init__()

        self.input_size = input_size
        self.hidden_units = hidden_units

    def forward(self, input_data, masks, state):
        raise NotImplementedError

    def initial_state(self):
        """ Initial state of the network """
        raise NotImplementedError
