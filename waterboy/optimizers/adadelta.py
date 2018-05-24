import torch.optim


def create():
    """ Return an ADADELTA optimizer """
    return torch.optim.Adadelta
