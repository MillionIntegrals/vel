import collections

from vel.api import SizeHints
from vel.net.layer_base import LayerFactory, Layer, LayerInfo, LayerFactoryContext
from vel.net.modular import LayerList


class SequenceLayer(Layer):
    """ Container around a skip connection """

    def __init__(self, info: LayerInfo, layers: [Layer]):
        super().__init__(info)

        self.layers = LayerList(layers)

    @property
    def is_stateful(self) -> bool:
        return self.layers.is_stateful

    def zero_state(self, batch_size):
        return self.layers.zero_state(batch_size)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self.layers[-1].size_hints()

    def forward(self, direct, state: dict = None, context: dict = None):
        """ Forward propagation of a single layer """
        return self.layers(direct, state=state, context=context)


class SequenceFactory(LayerFactory):
    """ Factory for skip connection layers """

    def __init__(self, layers: [LayerFactory]):
        super().__init__()
        self.layers = layers

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "skip_connection"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        loop_size_hints = direct_input

        layers = collections.OrderedDict()

        info = self.make_info(context)

        for idx,  layer_factory in enumerate(self.layers):
            counter = idx + 1

            child_context = LayerFactoryContext(
                idx=counter,
                parent_group=info.group,
                parent_name=info.name,
                data=context.data
            )

            layer = layer_factory.instantiate(
                direct_input=loop_size_hints,
                context=child_context,
                extra_args=extra_args
            )

            loop_size_hints = layer.size_hints()

            layers[layer.name] = layer

        return SequenceLayer(info, layers=layers)


def create(layers: [LayerFactory], label=None, group=None):
    """ Vel factory function """
    return SequenceFactory(layers=layers).with_given_name(label).with_given_group(group)
