import torch
import torch.utils.data

from torchvision import datasets, transforms


from waterboy.internals.source import Source


def create(batch_size, model_config, normalize=True, num_workers=0):
    """ Create a MNIST dataset, denormalized """
    kwargs = {}

    path = model_config.data_dir('cifar10')

    train_dataset = datasets.CIFAR10(path, train=True, download=True)
    test_dataset = datasets.CIFAR10(path, train=False, download=True)

    if normalize:
        train_data = train_dataset.train_data
        mean_value = (train_data / 255).mean()
        std_value = (train_data / 255).std()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean_value,), (std_value,))
        ])
    else:
        transform = transforms.ToTensor()

    train_dataset.transform = transform
    test_dataset.transform = transform

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs
    )

    return Source(train_loader, test_loader)
