import torch.optim

from vel.api.base import OptimizerFactory


class RMSpropFactory(OptimizerFactory):
    """ RMSprop optimizer factory """

    def __init__(self, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

    def instantiate(self, parameters) -> torch.optim.RMSprop:
        return torch.optim.RMSprop(
            parameters,
            lr=self.lr, alpha=self.alpha, eps=self.eps,
            weight_decay=self.weight_decay, momentum=self.momentum, centered=self.centered
        )


def create(lr, alpha, momentum=0, weight_decay=0, epsilon=1e-8):
    """ Vel creation function - RMSprop optimizer"""
    return RMSpropFactory(lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay, eps=float(epsilon))
