import torch.optim

from vel.api.base import OptimizerFactory


class AdamFactory(OptimizerFactory):
    """ Adam optimizer factory """

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def instantiate(self, parameters) -> torch.optim.Adam:
        return torch.optim.Adam(
            parameters,
            lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad
        )


def create(lr, weight_decay=0, epsilon=1e-8):
    """ Return an ADAM optimizer """
    return AdamFactory(lr=lr, weight_decay=weight_decay, eps=epsilon)
