from vel.api import Network
from vel.module.layers import OneHotEncode


class OneHotEncodingInput(Network):
    """ One-hot encoding input layer """

    def __init__(self, alphabet_size: int):
        super().__init__()

        self._alphabet_size = alphabet_size

        self.layer = OneHotEncode(self._alphabet_size)

    def forward(self, input_data):
        return self.layer(input_data)

