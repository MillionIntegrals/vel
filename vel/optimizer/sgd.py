import typing

from torch.optim.sgd import SGD

import vel.util.module_util as mu

from vel.api import OptimizerFactory, VelOptimizer, VelOptimizerProxy


class SgdFactory(OptimizerFactory):
    """ SGD optimizer factory """

    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 max_grad_norm: typing.Optional[float] = None):
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        return VelOptimizerProxy(
            SGD(
                parameters,
                lr=self.lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay,
                nesterov=self.nesterov
            ), self.max_grad_norm
        )

    def instantiate_parameter_groups(self, parameters) -> VelOptimizer:
        settings_dict = {
            'lr': self.lr,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov
        }

        parameters = parameters.copy()
        out_settings_dict = mu.optimizer_parameter_helper(parameters, settings_dict)

        return VelOptimizerProxy(SGD(parameters, **out_settings_dict), self.max_grad_norm)


def create(lr, momentum=0, dampening=0, weight_decay=0, nesterov=False,
           max_grad_norm: typing.Optional[float] = None):
    """ Vel factory function """
    return SgdFactory(
        lr=lr, momentum=momentum, dampening=dampening,
        weight_decay=weight_decay, nesterov=nesterov, max_grad_norm=max_grad_norm
    )
