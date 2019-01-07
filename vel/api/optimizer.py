from vel.api import Model

from torch.optim import Optimizer


class OptimizerFactory:
    """ Base class for optimizer factories """

    def instantiate(self, model: Model) -> Optimizer:
        raise NotImplementedError
