from vel.api import SizeHints
from vel.net.layer_base import Layer, LayerFactory, LayerInfo, LayerFactoryContext

from vel.util.tensor_util import one_hot_encoding


class AlphabetOneHotEncodeLayer(Layer):
    """
    Encode incoming tensor encoded using certain alphabet into one-hot encoding
    """
    def __init__(self, info: LayerInfo, alphabet_size: int, input_shape: SizeHints):
        super().__init__(info)

        self.alphabet_size = alphabet_size
        self.output_shape = SizeHints(input_shape.assert_single().append(self.alphabet_size + 1))

    def forward(self, direct, state: dict = None, context: dict = None):
        return one_hot_encoding(direct, num_labels=self.alphabet_size + 1)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.output_shape


class AlphabetOneHotEncodeLayerFactory(LayerFactory):
    """ Factory class for the AlphabetOneHotEncode layer """

    def __init__(self, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "alphabet_one_hot_encode"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        if 'alphabet_size' in extra_args:
            alphabet_size = extra_args['alphabet_size']
        else:
            alphabet_size = self.alphabet_size

        return AlphabetOneHotEncodeLayer(
            info=self.make_info(context),
            alphabet_size=alphabet_size,
            input_shape=direct_input
        )


def create(alphabet_size=None, label=None, group=None):
    """ Vel factory function """
    return AlphabetOneHotEncodeLayerFactory(alphabet_size=alphabet_size).with_given_name(label).with_given_group(group)
