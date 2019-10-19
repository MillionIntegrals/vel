"""
RAdam implementation from: https://github.com/LiyuanLucasLiu/RAdam/blob/master/cifar_imagenet/utils/radam.py
"""
import math
import torch
import typing

from torch.optim.optimizer import Optimizer

from vel.api import OptimizerFactory, VelOptimizer, VelOptimizerProxy


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class RAdamFactory(OptimizerFactory):
    """ Adam optimizer factory """

    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 max_grad_norm: typing.Optional[float] = None):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def instantiate(self, parameters) -> VelOptimizer:
        optimizer_params, group_names = self.preprocess(parameters)

        return VelOptimizerProxy(RAdam(
            optimizer_params,
            lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay
        ), group_names, self.max_grad_norm)


def create(lr, betas=(0.9, 0.999), weight_decay=0, epsilon=1e-8, max_grad_norm=None, parameter_groups=None):
    """ Vel factory function """
    return RAdamFactory(
        lr=lr, betas=betas, weight_decay=weight_decay, eps=epsilon, max_grad_norm=max_grad_norm
    ).with_parameter_groups(parameter_groups)
