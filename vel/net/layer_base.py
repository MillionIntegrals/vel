import attr
import typing

from vel.api import BackboneModule, SizeHints


@attr.s(auto_attribs=True)
class LayerInfo:
    """ Information about the layer """
    name: str
    global_name: str
    group: str


class Layer(BackboneModule):
    """ Layer class that fits into modular network framework """
    def __init__(self, info: LayerInfo):
        super().__init__()

        self.info = info

    @property
    def name(self) -> str:
        """ Name of this layer """
        return self.info.name

    @property
    def global_name(self) -> str:
        """ Name of this layer - globally unique version """
        return self.info.global_name

    @property
    def group(self) -> str:
        """ Group of this layer """
        return self.info.group

    def forward(self, direct, state: dict, context: dict):
        """ Forward propagation of a single layer """
        raise NotImplementedError

    def grouped_parameters(self):
        """ Return iterable of pairs (group, parameters) """
        return [(self.group, self.parameters())]


@attr.s(auto_attribs=True)
class LayerFactoryContext:
    """ Context information about the layer being currently created """

    idx: int
    """ Index of this layer within parent """

    parent_group: str
    """ Group of the parent layer """

    parent_name: typing.Optional[str] = None
    """ Name of the parent - None if it's a top level layer """

    data: dict = {}
    """ Generic information potentially passed by layer in a hierarchy """


class LayerFactory:
    """ Factory for layers """
    def __init__(self):
        self.given_name = None
        self.given_group = None

    def with_given_name(self, given_name) -> 'LayerFactory':
        """ Set given name """
        self.given_name = given_name
        return self

    def with_given_group(self, given_group) -> 'LayerFactory':
        """ Set given group """
        self.given_group = given_group
        return self

    def suggested_name(self, idx: int):
        """ Reasonable layer name suggestion """
        return "{}_{:04d}".format(self.name_base, idx)

    def make_info(self, context: LayerFactoryContext) -> LayerInfo:
        """ Make info for child layer """
        if self.given_name is not None:
            name = self.given_name
        else:
            name = self.suggested_name(context.idx)

        if self.given_group is not None:
            group = self.given_group
        else:
            group = context.parent_group

        if context.parent_name is None:
            global_name = name
        else:
            global_name = f"{context.parent_name}/{name}"

        return LayerInfo(
            name=name,
            group=group,
            global_name=global_name
        )

    @property
    def name_base(self) -> str:
        """ Base of layer name """
        raise NotImplementedError

    def instantiate(self, direct_input: SizeHints, context: LayerFactoryContext, extra_args: dict) -> Layer:
        """ Create a given layer object """
        raise NotImplementedError
