import typing

from torch.optim.adadelta import Adadelta

import vel.util.module_util as mu

from vel.api import OptimizerFactory, VelOptimizerProxy, VelOptimizer


class AdadeltaFactory(OptimizerFactory):
    """ Adadelta optimizer factory """

    def __init__(self, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6, weight_decay: float = 0.0,
                 max_grad_norm: typing.Optional[float] = None):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        return VelOptimizerProxy(Adadelta(
            parameters,
            lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay
        ), self.max_grad_norm)

    def instantiate_parameter_groups(self, out_parameters) -> VelOptimizer:
        settings_dict = {
            'lr': self.lr,
            'rho': self.rho,
            'eps': self.eps,
            'weight_decay': self.weight_decay
        }

        out_parameters = out_parameters.copy()
        out_settings_dict = mu.optimizer_parameter_helper(out_parameters, settings_dict)

        return VelOptimizerProxy(Adadelta(out_parameters, **out_settings_dict), self.max_grad_norm)


def create(lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6, weight_decay: float = 0.0,
           max_grad_norm: typing.Optional[float] = None):
    """ Vel factory function """
    return AdadeltaFactory(
        lr=lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm
    )
