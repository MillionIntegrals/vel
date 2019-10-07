import torch.nn.functional as F
from vel.api import SizeHints
from vel.net.layer_base import Layer, LayerFactory


class DropoutLayer(Layer):
    """
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.
    """
    def __init__(self, name: str, input_size: SizeHints, p: float):
        super().__init__(name)

        self.p = p
        self.input_size = input_size

    def forward(self, direct, state: dict = None, context: dict = None):
        return F.dropout(direct, p=self.p, training=self.training)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.input_size

    def extra_repr(self) -> str:
        """Set the extra representation of the module"""
        return "p={:.2f}".format(self.p)


class DropoutLayerFactory(LayerFactory):
    """ Factory class for the Dropout layer """

    def __init__(self, p: float):
        self.p = p

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "dropout"

    def instantiate(self, name: str, direct_input: SizeHints, context: dict, extra_args: dict) -> Layer:
        """ Create a given layer object """
        return DropoutLayer(
            name=name,
            input_size=direct_input,
            p=self.p
        )


def create(p: float):
    """ Vel factory function """
    return DropoutLayerFactory(p)
