import collections

from vel.api import SizeHints, SizeHint
from vel.net.layer_base import LayerFactory, Layer, LayerInfo, LayerFactoryContext
from vel.net.modular import LayerList


class SkipConnectionLayer(Layer):
    """ Container around a skip connection """

    def __init__(self, info: LayerInfo, layers: [Layer], size_hint: SizeHint):
        super().__init__(info)

        self.layers = LayerList(layers)
        self._size_hints = SizeHints(size_hint)

    @property
    def is_stateful(self) -> bool:
        return self.layers.is_stateful

    def zero_state(self, batch_size):
        return self.layers.zero_state(batch_size)

    def size_hints(self) -> SizeHints:
        """ Size hints for this network """
        return self._size_hints

    def forward(self, direct, state: dict = None, context: dict = None):
        """ Forward propagation of a single layer """
        if self.is_stateful:
            result, out_state = self.layers(direct, state=state, context=context)
            return direct + result, out_state
        else:
            result = self.layers(direct, state=state, context=context)
            return direct + result


class SkipConnectionLayerFactory(LayerFactory):
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
        size_hint = direct_input.assert_single()

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
                direct_input=SizeHints(size_hint),
                context=child_context,
                extra_args=extra_args
            )

            layers[layer.name] = layer

        return SkipConnectionLayer(info, layers=layers, size_hint=size_hint)


def create(layers: [LayerFactory], label=None, group=None):
    """ Vel factory function """
    return SkipConnectionLayerFactory(layers=layers).with_given_name(label).with_given_group(group)
