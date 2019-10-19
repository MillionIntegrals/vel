import typing

from torch.optim.adadelta import Adadelta

from vel.api import OptimizerFactory, VelOptimizerProxy, VelOptimizer


class AdadeltaFactory(OptimizerFactory):
    """ Adadelta optimizer factory """

    def __init__(self, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6, weight_decay: float = 0.0,
                 max_grad_norm: typing.Optional[float] = None):
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        optimizer_params, group_names = self.preprocess(parameters)

        return VelOptimizerProxy(Adadelta(
            optimizer_params,
            lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay
        ), group_names, self.max_grad_norm)


def create(lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6, weight_decay: float = 0.0,
           max_grad_norm: typing.Optional[float] = None, parameter_groups=None):
    """ Vel factory function """
    return AdadeltaFactory(
        lr=lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm
    ).with_parameter_groups(parameter_groups)
