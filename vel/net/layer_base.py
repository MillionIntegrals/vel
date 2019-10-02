import typing

from vel.api import BackboneNetwork, SizeHints, SizeHint


class Layer(BackboneNetwork):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def state_size_hints(self) -> typing.Dict[str, SizeHint]:
        """ Size hints for state part of this network """
        return {}

    def forward(self, direct, state: dict = None, context: dict = None):
        """ Forward propagation of a single layer """
        raise NotImplementedError


class LayerFactory:
    """ Factory for layers """

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        raise NotImplementedError

    def instantiate(self, name: str, direct_input: SizeHints, context: dict) -> Layer:
        """ Create a given layer object """
        raise NotImplementedError

