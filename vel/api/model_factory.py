from .model import Model
from vel.internals.generic_factory import GenericFactory


class ModelFactory:
    """ Factory class for models """

    def instantiate(self, **extra_args) -> Model:
        raise NotImplementedError

    @staticmethod
    def generic(closure, **kwargs) -> 'ModelFactory':
        """ Return a generic model factory """
        # noinspection PyTypeChecker
        return GenericFactory(closure, kwargs)
