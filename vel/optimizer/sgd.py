import typing

from torch.optim.sgd import SGD

from vel.api import OptimizerFactory, VelOptimizer, VelOptimizerProxy


class SgdFactory(OptimizerFactory):
    """ SGD optimizer factory """

    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 max_grad_norm: typing.Optional[float] = None):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        optimizer_params, group_names = self.preprocess(parameters)

        return VelOptimizerProxy(SGD(
            optimizer_params,
            lr=self.lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay,
            nesterov=self.nesterov
        ), group_names, self.max_grad_norm)


def create(lr, momentum=0, dampening=0, weight_decay=0, nesterov=False,
           max_grad_norm: typing.Optional[float] = None, parameter_groups=None):
    """ Vel factory function """
    return SgdFactory(
        lr=lr, momentum=momentum, dampening=dampening,
        weight_decay=weight_decay, nesterov=nesterov, max_grad_norm=max_grad_norm
    ).with_parameter_groups(parameter_groups)
