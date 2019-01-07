import torch.nn as nn

from vel.api import LinearBackboneModel, TextData, ModelFactory


class EmbeddingInput(LinearBackboneModel):
    """ Learnable Embedding input layer """

    def __init__(self, alphabet_size: int, output_dim: int, pretrained: bool=False, frozen: bool=False,
                 source: TextData=None):
        super().__init__()

        self._output_dim = output_dim
        self._alphabet_size = alphabet_size
        self._pretrained = pretrained
        self._frozen = frozen
        self._source = source

        self.layer = nn.Embedding(self._alphabet_size, self._output_dim)

    def reset_weights(self):
        if self._pretrained:
            self.layer.weight.data.copy_(self._source.data_field.vocab.vectors)

        if self._frozen:
            self.layer.weight.requires_grad = False

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self._output_dim

    def forward(self, input_data):
        return self.layer(input_data)


def create(alphabet_size: int, output_dim: int, pretrained: bool=False, frozen: bool=False, source: TextData=None):
    """ Vel factory function """
    def instantiate(**_):
        return EmbeddingInput(alphabet_size, output_dim, pretrained=pretrained, frozen=frozen, source=source)

    return ModelFactory.generic(instantiate)


# Scripting interface
EmbeddingInputFactory = create

