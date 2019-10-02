from vel.module.layers import Flatten


from vel.api import ModelFactory, BackboneModel


class FlattenInput(BackboneModel):
    """ Sequence input """
    def __init__(self):
        super().__init__()
        self.model = Flatten()

    def forward(self, input_data):
        return self.model(input_data)


def create():
    """ Vel factory function """
    def instantiate(**_):
        return Flatten()

    return ModelFactory.generic(instantiate)
