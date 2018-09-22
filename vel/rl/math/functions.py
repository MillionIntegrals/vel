import torch


def explained_variance(returns, values):
    """ Calculate how much variance in returns do the values explain """
    exp_var = 1 - torch.var(returns - values) / torch.var(returns)
    return exp_var.item()
