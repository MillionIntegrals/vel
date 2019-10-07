from vel.internal.generic_factory import GenericFactory
from .vmodule import VModule


class ModuleFactory:
    """ Factory for modules """

    def instantiate(self, **extra_args) -> VModule:
        raise NotImplementedError

    @staticmethod
    def generic(closure, **kwargs) -> 'ModuleFactory':
        """ Return a generic model factory """
        # noinspection PyTypeChecker
        return GenericFactory(closure, kwargs)
