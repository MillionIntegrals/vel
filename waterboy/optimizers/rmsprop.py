import torch.optim as opt


def create(lr, alpha, momentum=0, weight_decay=0):
    """ Waterboy creation function - RMSprop optimizer"""
    def optimizer_fn(params):
        return opt.RMSprop(params, lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay)

    return optimizer_fn
