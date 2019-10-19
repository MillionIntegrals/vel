from vel.api import VModule
from vel.module.layers import OneHotEncode


class OneHotEncodingInput(VModule):
    """ One-hot encoding input layer """

    def __init__(self, alphabet_size: int):
        super().__init__()

        self._alphabet_size = alphabet_size

        self.layer = OneHotEncode(self._alphabet_size)

    def forward(self, input_data):
        return self.layer(input_data)
