import torch.optim

from vel.api.base import OptimizerFactory


class AdadeltaFactory(OptimizerFactory):
    """ Adadelta optimizer factory """

    def __init__(self, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

    def instantiate(self, parameters) -> torch.optim.Adadelta:
        return torch.optim.Adadelta(
            parameters,
            lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay
        )


def create():
    """ Return an ADADELTA optimizer """
    return AdadeltaFactory()
