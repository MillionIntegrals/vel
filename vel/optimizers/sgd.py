import torch.optim

import vel.util.module_util as mu

from vel.api import OptimizerFactory, Model


class SgdFactory(OptimizerFactory):
    """ SGD optimizer factory """

    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, layer_groups: bool=False):
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.layer_groups = layer_groups

    def instantiate(self, model: Model) -> torch.optim.SGD:
        if self.layer_groups:
            parameters = mu.to_parameter_groups(model.get_layer_groups())
        else:
            parameters = filter(lambda p: p.requires_grad, model.parameters())

        return torch.optim.SGD(
            parameters,
            lr=self.lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay,
            nesterov=self.nesterov
        )


def create(lr, weight_decay=0, momentum=0, layer_groups=False):
    """ Vel factory function """
    return SgdFactory(lr=lr, weight_decay=weight_decay, momentum=momentum, layer_groups=layer_groups)
