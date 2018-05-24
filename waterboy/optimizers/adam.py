import torch.optim


def create(lr, weight_decay=0):
    """ Return an ADAM optimizer """
    def optimizer_fn(params):
        return torch.optim.Adam(params, lr, weight_decay=weight_decay)

    return optimizer_fn
