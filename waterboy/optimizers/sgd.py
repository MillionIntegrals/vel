import torch.optim


def create(lr, weight_decay=0, momentum=0):
    """ Return an SGD optimizer """
    def optimizer_fn(params):
        return torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer_fn
