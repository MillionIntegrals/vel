from .model import Model


class ModelAugmentor:
    """ Factory base class that instantiates models that need some extra information """

    def augment(self, base_model: Model, extra_info: dict=None) -> Model:
        """ Instantiate model """
        raise NotImplementedError
