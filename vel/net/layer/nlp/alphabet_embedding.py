import torch.nn as nn

from vel.api import SizeHints
from vel.net.layer_base import Layer, LayerFactory


class AlphabetEmbeddingLayer(Layer):
    """
    Encode incoming tensor encoded using certain alphabet into one-hot encoding
    """
    def __init__(self, name: str, alphabet_size: int, dim: int, input_shape: SizeHints):
        super().__init__(name)

        self.alphabet_size = alphabet_size
        self.dim = dim
        self.output_size = SizeHints(input_shape.assert_single().append(self.dim))

        self.layer = nn.Embedding(self.alphabet_size + 1, self.dim)

    def forward(self, direct, state: dict = None, context: dict = None):
        return self.layer(direct)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.output_size


class AlphabetEmbeddingLayerFactory(LayerFactory):
    """ Factory class for the AlphabetOneHotEncode layer """

    def __init__(self, dim: int):
        self.dim = dim

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "alphabet_embedding"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict, extra_args: dict) -> Layer:
        alphabet_size = extra_args['alphabet_size']

        return AlphabetEmbeddingLayer(
            name=name,
            alphabet_size=alphabet_size,
            dim=self.dim,
            input_shape=direct_input
        )


def create(dim: int):
    """ Vel factory function """
    return AlphabetEmbeddingLayerFactory(dim)
