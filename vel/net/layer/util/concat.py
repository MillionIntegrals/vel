import torch

from vel.api import SizeHints, SizeHint
from vel.net.layer_base import LayerFactory, Layer


class Concat(Layer):
    """ Repeat single tensor multiple times """

    def __init__(self, name: str, size_hints: SizeHints, axis: int = -1):
        super().__init__(name)

        self.axis = axis
        self._size_hints = size_hints

    def forward(self, direct, state: dict = None, context: dict = None):
        return torch.cat(direct, dim=self.axis)

    def size_hints(self) -> SizeHints:
        return self._size_hints


class ConcatFactory(LayerFactory):
    def __init__(self, axis: int = -1):
        self.axis = axis

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "concat"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict) -> Layer:
        inputs = direct_input.assert_tuple()

        result = []
        dimension = len(inputs[0])

        for i in range(dimension):

            if i == (self.axis % dimension):
                candidates = [el[i] for el in inputs]

                if None in candidates:
                    result.append(None)
                else:
                    result.append(sum(candidates))
            else:
                result.append(inputs[0][i])

        return Concat(
            name=name,
            axis=self.axis,
            size_hints=SizeHints(SizeHint(*result))
        )


def create(axis: int = -1):
    """ Vel factory function """
    return ConcatFactory(axis=axis)
