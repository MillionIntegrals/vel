import torch.nn.functional as F
from vel.api import SizeHints
from vel.net.layer_base import Layer, LayerFactory, LayerFactoryContext, LayerInfo


class DropoutLayer(Layer):
    """
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.
    """
    def __init__(self, info: LayerInfo, input_shape: SizeHints, p: float):
        super().__init__(info)

        self.p = p
        self.input_shape = input_shape

    def forward(self, direct, state: dict = None, context: dict = None):
        return F.dropout(direct, p=self.p, training=self.training)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.input_shape

    def extra_repr(self) -> str:
        """Set the extra representation of the module"""
        return "p={:.2f}".format(self.p)


class DropoutLayerFactory(LayerFactory):
    """ Factory class for the Dropout layer """

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "dropout"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        return DropoutLayer(
            info=self.make_info(context),
            input_shape=direct_input,
            p=self.p
        )


def create(p: float, label=None, group=None):
    """ Vel factory function """
    return DropoutLayerFactory(p).with_given_name(label).with_given_group(group)
