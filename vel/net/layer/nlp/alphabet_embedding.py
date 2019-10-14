import torch.nn as nn

from vel.api import SizeHints
from vel.net.layer_base import Layer, LayerFactory, LayerFactoryContext, LayerInfo


class AlphabetEmbeddingLayer(Layer):
    """
    Encode incoming tensor encoded using certain alphabet into one-hot encoding
    """
    def __init__(self, info: LayerInfo, alphabet_size: int, dim: int, input_shape: SizeHints):
        super().__init__(info)

        self.alphabet_size = alphabet_size
        self.dim = dim
        self.output_shape = SizeHints(input_shape.assert_single().append(self.dim))

        self.layer = nn.Embedding(self.alphabet_size + 1, self.dim)

    def forward(self, direct, state: dict = None, context: dict = None):
        """ Forward propagation of a single layer """
        return self.layer(direct)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.output_shape


class AlphabetEmbeddingLayerFactory(LayerFactory):
    """ Factory class for the AlphabetOneHotEncode layer """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "alphabet_embedding"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        alphabet_size = extra_args['alphabet_size']

        return AlphabetEmbeddingLayer(
            info=self.make_info(context),
            alphabet_size=alphabet_size,
            dim=self.dim,
            input_shape=direct_input
        )


def create(dim: int, label=None, group=None):
    """ Vel factory function """
    return AlphabetEmbeddingLayerFactory(dim).with_given_name(label).with_given_group(group)
