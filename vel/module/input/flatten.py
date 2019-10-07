from vel.module.layers import Flatten


from vel.api import Network


class FlattenInput(Network):
    """ Sequence input """
    def __init__(self):
        super().__init__()
        self.model = Flatten()

    def forward(self, input_data):
        return self.model(input_data)

