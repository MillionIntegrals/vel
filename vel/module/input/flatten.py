from vel.module.layers import Flatten


from vel.api import ModelFactory, BackboneNetwork


class FlattenInput(BackboneNetwork):
    """ Sequence input """
    def __init__(self):
        super().__init__()
        self.model = Flatten()

    def forward(self, input_data):
        return self.model(input_data)

