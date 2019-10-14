import torch

from vel.api import SizeHints, SizeHint
from vel.net.layer_base import Layer, LayerFactory, LayerInfo, LayerFactoryContext


class SelectFinalFeaturesLayer(Layer):
    """
    For many sequence processing tasks we only care about the output from the final element
    """
    def __init__(self, info: LayerInfo, bidirectional: bool, input_shape: SizeHints):
        super().__init__(info)

        self.bidirectional = bidirectional

        b, s, x = input_shape.assert_single(3)
        self.output_shape = SizeHints(SizeHint(b, x))

    def forward(self, direct, state: dict = None, context: dict = None):
        if self.bidirectional:
            final_shape = direct.shape[-1]
            assert final_shape % 2 == 0
            half_final_shape = final_shape // 2

            # dimensions are: batch, seq, features
            # first one is from forward pass
            # second one is backward pass
            part1 = direct[:, -1, :half_final_shape]
            part2 = direct[:, 0, half_final_shape:]

            return torch.cat([part1, part2], dim=1)
        else:
            return direct[:, -1, :]

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.output_shape


class SelectFinalFeaturesLayerFactory(LayerFactory):
    """ Factory for the SelectFinalFeatures layer """

    def __init__(self, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "select_final_features"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        return SelectFinalFeaturesLayer(
            info=self.make_info(context),
            bidirectional=self.bidirectional,
            input_shape=direct_input
        )


def create(bidirectional=False, label=None, group=None):
    """ Vel factory function """
    return SelectFinalFeaturesLayerFactory(
        bidirectional=bidirectional
    ).with_given_name(label).with_given_group(group)
