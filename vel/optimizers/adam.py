import collections
import torch.optim

import vel.util.module_util as mu

from vel.api import OptimizerFactory, Model


class AdamFactory(OptimizerFactory):
    """ Adam optimizer factory """

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, layer_groups=False):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.layer_groups = layer_groups

    def instantiate(self, model: Model) -> torch.optim.Adam:
        if self.layer_groups:
            parameters = mu.to_parameter_groups(model.get_layer_groups())

            if isinstance(self.lr, collections.Sequence):
                for idx, lr in enumerate(self.lr):
                    parameters[idx]['lr'] = lr

                default_lr = self.lr[0]
            else:
                default_lr = float(self.lr)

            if isinstance(self.weight_decay, collections.Sequence):
                for idx, weight_decay in enumerate(self.weight_decay):
                    parameters[idx]['weight_decay'] = weight_decay

                default_weight_decay = self.weight_decay[0]
            else:
                default_weight_decay = self.weight_decay

            return torch.optim.Adam(
                parameters,
                lr=default_lr, betas=self.betas, eps=self.eps, weight_decay=default_weight_decay, amsgrad=self.amsgrad
            )
        else:
            parameters = filter(lambda p: p.requires_grad, model.parameters())

            return torch.optim.Adam(
                parameters,
                lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad
            )


def create(lr, betas=(0.9, 0.999), weight_decay=0, epsilon=1e-8, layer_groups=False):
    """ Vel factory function """
    return AdamFactory(lr=lr, betas=betas, weight_decay=weight_decay, eps=epsilon, layer_groups=layer_groups)
