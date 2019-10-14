import typing

from torch.optim.rmsprop import RMSprop

from vel.api import OptimizerFactory, VelOptimizerProxy, VelOptimizer


class RMSpropFactory(OptimizerFactory):
    """ RMSprop optimizer factory """

    def __init__(self, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
                 max_grad_norm: typing.Optional[float] = None):
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        optimizer_params, group_names = self.preprocess(parameters)

        return VelOptimizerProxy(RMSprop(
            optimizer_params,
            lr=self.lr, alpha=self.alpha, eps=self.eps,
            weight_decay=self.weight_decay, momentum=self.momentum, centered=self.centered
        ), group_names, self.max_grad_norm)


def create(lr, alpha, momentum=0, weight_decay=0, epsilon=1e-8, max_grad_norm=None, parameter_groups=None):
    """ Vel factory function """
    return RMSpropFactory(
        lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay, eps=float(epsilon),
        max_grad_norm=max_grad_norm
    ).with_parameter_groups(parameter_groups)
