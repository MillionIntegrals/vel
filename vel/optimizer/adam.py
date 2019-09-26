import typing

from torch.optim.adam import Adam

import vel.util.module_util as mu

from vel.api import OptimizerFactory, VelOptimizer, VelOptimizerProxy


class AdamFactory(OptimizerFactory):
    """ Adam optimizer factory """

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
                 max_grad_norm: typing.Optional[float] = None):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        return VelOptimizerProxy(Adam(
            parameters,
            lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad
        ), self.max_grad_norm)

    def instantiate_parameter_groups(self, out_parameters) -> VelOptimizer:
        settings_dict = {
            'lr': self.lr,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad
        }

        out_parameters = out_parameters.copy()
        out_settings_dict = mu.optimizer_parameter_helper(out_parameters, settings_dict)

        return VelOptimizerProxy(Adam(out_parameters, betas=self.betas, **out_settings_dict), self.max_grad_norm)


def create(lr, betas=(0.9, 0.999), weight_decay=0, epsilon=1e-8, max_grad_norm=None):
    """ Vel factory function """
    return AdamFactory(lr=lr, betas=betas, weight_decay=weight_decay, eps=epsilon, max_grad_norm=max_grad_norm)
