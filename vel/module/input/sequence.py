import torch.nn as nn

from vel.api import ModelFactory, BackboneModel


class SequenceInput(BackboneModel):
    """ Sequence input """
    def __init__(self, modules):
        super().__init__()
        self.model = nn.Sequential(*modules)

    def forward(self, input_data):
        return self.model(input_data)


def create(modules):
    """ Vel factory function """
    def instantiate(**_):
        return SequenceInput([f.instantiate() for f in modules])

    return ModelFactory.generic(instantiate)
