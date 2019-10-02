import typing

from vel.api import SizeHints, SizeHint
from vel.net.modular import LayerFactory, Layer


class RepeatTensor(Layer):
    """ Repeat single tensor multiple times """

    def __init__(self, name: str, times: int, size_hint: SizeHint):
        super().__init__(name)
        self.times = times
        self.size_hint = size_hint

    def forward(self, direct, state: dict = None, context: dict = None):
        return tuple([direct] * self.times)

    def size_hints(self) -> SizeHints:
        return SizeHints(tuple([self.size_hint] * self.times))


class RepeatTensorFactory(LayerFactory):
    def __init__(self, times: int):
        self.times = times

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "repeat_tensor"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict) -> Layer:
        return RepeatTensor(
            name=name,
            times=self.times,
            size_hint=direct_input.assert_single()
        )


def create(times: int):
    """ Vel factory function """
    return RepeatTensorFactory(times=times)
