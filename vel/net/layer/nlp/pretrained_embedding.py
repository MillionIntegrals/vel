import numpy as np

import torch.nn as nn

from vel.api import SizeHints, LanguageSource
from vel.net.layer_base import Layer, LayerFactory, LayerFactoryContext, LayerInfo


class PretrainedEmbeddingLayer(Layer):
    """ Load a pretrained word embedding """
    def __init__(self, info: LayerInfo, vectors: np.ndarray, input_shape: SizeHints, freeze: bool = False):
        super().__init__(info)

        self.output_shape = SizeHints(input_shape.assert_single().append(vectors.shape[1]))

        self.layer = nn.Embedding(vectors.shape[0], vectors.shape[1])
        self.layer.weight.data.copy_(vectors)

        self.freeze = freeze

        if self.freeze:
            self.layer.weight.requires_grad_(False)

    def forward(self, direct, state: dict = None, context: dict = None):
        return self.layer(direct)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.output_shape


class PretrainedEmbeddingLayerFactory(LayerFactory):
    """ Load a pretrained word embedding """
    def __init__(self, source: LanguageSource, vectors: str, freeze: bool):
        super().__init__()
        self.vectors = vectors
        self.source = source
        self.freeze = freeze

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "pretrained_embedding"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        vocab = self.source.fields[self.source.mapping['x']].vocab
        vocab.load_vectors(self.vectors)

        return PretrainedEmbeddingLayer(
            info=self.make_info(context),
            vectors=vocab.vectors,
            freeze=self.freeze,
            input_shape=direct_input,
        )


def create(source: LanguageSource, vectors: str, freeze: bool = False, label=None, group=None):
    """ Vel factory function """
    return PretrainedEmbeddingLayerFactory(
        source, vectors, freeze=freeze
    ).with_given_name(label).with_given_group(group)
