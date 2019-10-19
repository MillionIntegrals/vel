from vel.api import SizeHints, SizeHint
from vel.net.layer_base import LayerFactory, Layer, LayerInfo, LayerFactoryContext


class RepeatTensor(Layer):
    """ Repeat single tensor multiple times """

    def __init__(self, info: LayerInfo, times: int, size_hint: SizeHint):
        super().__init__(info)

        self.times = times
        self.size_hint = size_hint

    def forward(self, direct, state: dict = None, context: dict = None):
        return tuple([direct] * self.times)

    def size_hints(self) -> SizeHints:
        return SizeHints(tuple([self.size_hint] * self.times))


class RepeatTensorFactory(LayerFactory):
    def __init__(self, times: int):
        super().__init__()
        self.times = times

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "repeat_tensor"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        return RepeatTensor(
            info=self.make_info(context),
            times=self.times,
            size_hint=direct_input.assert_single()
        )


def create(times: int, label=None, group=None):
    """ Vel factory function """
    return RepeatTensorFactory(times=times).with_given_name(label).with_given_group(group)
