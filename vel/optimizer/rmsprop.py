import typing

from torch.optim.rmsprop import RMSprop

import vel.util.module_util as mu

from vel.api import OptimizerFactory, VelOptimizerProxy, VelOptimizer


class RMSpropFactory(OptimizerFactory):
    """ RMSprop optimizer factory """

    def __init__(self, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
                 max_grad_norm: typing.Optional[float] = None):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        return VelOptimizerProxy(RMSprop(
            parameters,
            lr=self.lr, alpha=self.alpha, eps=self.eps,
            weight_decay=self.weight_decay, momentum=self.momentum, centered=self.centered
        ), self.max_grad_norm)

    def instantiate_parameter_groups(self, out_parameters) -> VelOptimizer:
        settings_dict = {
            'lr': self.lr,
            'alpha': self.alpha,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'centered': self.centered
        }

        out_parameters = out_parameters.copy()
        out_settings_dict = mu.optimizer_parameter_helper(out_parameters, settings_dict)

        return VelOptimizerProxy(RMSprop(out_parameters, **out_settings_dict), self.max_grad_norm)


def create(lr, alpha, momentum=0, weight_decay=0, epsilon=1e-8, max_grad_norm=None):
    """ Vel factory function """
    return RMSpropFactory(
        lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay, eps=float(epsilon),
        max_grad_norm=max_grad_norm
    )
