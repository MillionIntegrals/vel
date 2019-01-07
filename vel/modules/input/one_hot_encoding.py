from vel.api import LinearBackboneModel, ModelFactory
from vel.modules.layers import OneHotEncode


class OneHotEncodingInput(LinearBackboneModel):
    """ One-hot encoding input layer """

    def __init__(self, alphabet_size: int):
        super().__init__()

        self._alphabet_size = alphabet_size

        self.layer = OneHotEncode(self._alphabet_size)

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self._alphabet_size

    def forward(self, input_data):
        return self.layer(input_data)


def create(alphabet_size: int):
    """ Vel factory function """
    def instantiate(**_):
        return OneHotEncodingInput(alphabet_size)

    return ModelFactory.generic(instantiate)


# Scripting interface
OneHotEncodingInputFactory = create
