import typing

from torch.optim.adam import Adam

from vel.api import OptimizerFactory, VelOptimizer, VelOptimizerProxy


class AdamFactory(OptimizerFactory):
    """ Adam optimizer factory """

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
                 max_grad_norm: typing.Optional[float] = None):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        optimizer_params, group_names = self.preprocess(parameters)

        return VelOptimizerProxy(Adam(
            optimizer_params,
            lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad
        ), group_names, self.max_grad_norm)


def create(lr, betas=(0.9, 0.999), weight_decay=0, epsilon=1e-8, max_grad_norm=None, parameter_groups=None):
    """ Vel factory function """
    return AdamFactory(
        lr=lr, betas=betas, weight_decay=weight_decay, eps=epsilon, max_grad_norm=max_grad_norm,
    ).with_parameter_groups(parameter_groups)
