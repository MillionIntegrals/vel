import torch.optim


def create(model):
    """ Return an ADADELTA optimizer """
    return torch.optim.Adadelta(model.parameters())
