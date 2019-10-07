from vel.api import SizeHints
from vel.net.layer_base import Layer, LayerFactory

from vel.util.tensor_util import one_hot_encoding


class AlphabetOneHotEncodeLayer(Layer):
    """
    Encode incoming tensor encoded using certain alphabet into one-hot encoding
    """
    def __init__(self, name: str, alphabet_size: int, input_shape: SizeHints):
        super().__init__(name)

        self.alphabet_size = alphabet_size
        self.output_size = SizeHints(input_shape.assert_single().append(self.alphabet_size + 1))

    def forward(self, direct, state: dict = None, context: dict = None):
        return one_hot_encoding(direct, num_labels=self.alphabet_size + 1)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.output_size


class AlphabetOneHotEncodeLayerFactory(LayerFactory):
    """ Factory class for the AlphabetoneHotEncode layer """

    def __init__(self):
        pass

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "alphabet_one_hot_encode"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict, extra_args: dict) -> Layer:
        alphabet_size = extra_args['alphabet_size']
        return AlphabetOneHotEncodeLayer(
            name=name,
            alphabet_size=alphabet_size,
            input_shape=direct_input
        )


def create():
    """ Vel factory function """
    return AlphabetOneHotEncodeLayerFactory()
