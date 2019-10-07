from vel.module.layers import Flatten


from vel.api import VModule


class FlattenInput(VModule):
    """ Sequence input """
    def __init__(self):
        super().__init__()
        self.model = Flatten()

    def forward(self, input_data):
        return self.model(input_data)

