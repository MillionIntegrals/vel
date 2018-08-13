from .model import Model


class ModelFactory:
    """ Factory class for models """

    def instantiate(self, **extra_args) -> Model:
        raise NotImplementedError

    @staticmethod
    def generic(closure):
        return GenericModelFactory(closure)


class GenericModelFactory(ModelFactory):
    """ Create model from a lambda function """
    def __init__(self, closure):
        self.closure = closure

    def instantiate(self, **extra_args):
        return self.closure(**extra_args)
