import torch.optim

from vel.api import OptimizerFactory, Model


class AdadeltaFactory(OptimizerFactory):
    """ Adadelta optimizer factory """

    def __init__(self, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

    def instantiate(self, model: Model) -> torch.optim.Adadelta:
        return torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay
        )


def create():
    """ Vel factory function """
    return AdadeltaFactory()
