import torch.optim

from vel.api.base import OptimizerFactory


class SgdFactory(OptimizerFactory):
    """ SGD optimizer factory """

    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def instantiate(self, parameters) -> torch.optim.SGD:
        return torch.optim.SGD(
            parameters,
            lr=self.lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay,
            nesterov=self.nesterov
        )


def create(lr, weight_decay=0, momentum=0):
    """ Return an SGD optimizer """
    return SgdFactory(lr=lr, weight_decay=weight_decay, momentum=momentum)
