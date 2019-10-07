from vel.api import BackboneNetwork, SizeHints


class Layer(BackboneNetwork):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, direct, state: dict = None, context: dict = None):
        """ Forward propagation of a single layer """
        raise NotImplementedError


class LayerFactory:
    """ Factory for layers """

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        raise NotImplementedError

    def instantiate(self, name: str, direct_input: SizeHints, context: dict, extra_args: dict) -> Layer:
        """ Create a given layer object """
        raise NotImplementedError

