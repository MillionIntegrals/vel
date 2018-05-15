import torch
from torchvision import datasets, transforms


class Source:
    def __init__(self, train_source, val_source, train_steps=None, val_steps=None,
                 x_train=None, y_train=None, x_val=None, y_val=None):
        self.train_source = train_source
        self.val_source = val_source

        # Optional number of steps per epoch
        self.train_steps = train_steps
        self.val_steps = val_steps

        # Optional, actual data points
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val



def create(batch_size):
    kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),

        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return Source(train_loader, test_loader)
