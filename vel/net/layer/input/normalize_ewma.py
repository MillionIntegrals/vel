from vel.api import SizeHints, SizeHint
from vel.module.input.normalize_ewma import NormalizeEwma
from vel.net.layer_base import LayerFactory, Layer, LayerFactoryContext, LayerInfo


class NormalizeEwmaLayer(Layer):
    """ Layer that normalizes the inputs """

    def __init__(self, info: LayerInfo, input_shape: SizeHints, beta: float = 0.99, epsilon: float = 1e-1,
                 per_element_update=False):
        super().__init__(info)

        self.input_shape = input_shape
        self.beta = beta
        self.epsilon = epsilon
        self.per_element_update = per_element_update

        self.normalize = NormalizeEwma(
            beta=self.beta,
            epsilon=self.epsilon,
            per_element_update=self.per_element_update,
            input_shape=self.input_shape.assert_single()[1:]  # Remove batch axis
        )

    def reset_weights(self):
        self.normalize.reset_weights()

    def forward(self, direct, state: dict = None, context: dict = None):
        return self.normalize(direct)

    def size_hints(self) -> SizeHints:
        return self.input_shape


class NormalizeEwmaLayerFactory(LayerFactory):
    def __init__(self, beta: float = 0.99, epsilon: float = 1e-2, shape=None, per_element_update=False):
        super().__init__()
        self.shape = shape
        self.beta = beta
        self.epsilon = epsilon
        self.per_element_update = per_element_update

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        return "image_to_tensor"

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        if self.shape is None:
            input_shape = direct_input
        else:
            input_shape = SizeHints(SizeHint(*([None] + list(self.shape))))

        return NormalizeEwmaLayer(
            info=self.make_info(context),
            beta=self.beta,
            epsilon=self.epsilon,
            per_element_update=self.per_element_update,
            input_shape=input_shape
        )


def create(beta=0.99, epsilon=1e-1, shape=None, per_element_update=False, label=None, group=None):
    """ Vel factory function """
    return NormalizeEwmaLayerFactory(
        beta=beta, epsilon=epsilon, shape=shape, per_element_update=per_element_update
    ).with_given_name(label).with_given_group(group)
