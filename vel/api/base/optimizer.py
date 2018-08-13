from torch.optim import Optimizer


class OptimizerFactory:
    """ Base class for optimizer factories """

    def instantiate(self, parameters) -> Optimizer:
        raise NotImplementedError
