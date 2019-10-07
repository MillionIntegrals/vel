from .vmodule import VModule
from vel.internal.generic_factory import GenericFactory


class ModelFactory:
    """ Factory class for models """

    def instantiate(self, **extra_args) -> VModule:
        raise NotImplementedError

    @staticmethod
    def generic(closure, **kwargs) -> 'ModelFactory':
        """ Return a generic model factory """
        # noinspection PyTypeChecker
        return GenericFactory(closure, kwargs)
