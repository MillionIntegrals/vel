import torch
import torch.utils.data

from torchvision import datasets, transforms


from waterboy.api.source import Source


def create(batch_size, model_config, normalize=True, num_workers=0):
    """ Create a MNIST dataset, normalized """
    kwargs = {}

    path = model_config.data_dir('mnist')

    train_dataset = datasets.MNIST(path, train=True, download=True)
    test_dataset = datasets.MNIST(path, train=False, download=True)

    if normalize:
        train_data = train_dataset.train_data
        mean_value = (train_data.double() / 255).mean().item()
        std_value = (train_data.double() / 255).std().item()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean_value,), (std_value,))
        ])
    else:
        transform = transforms.ToTensor()

    train_dataset.transform = transform
    test_dataset.transform = transform

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        **kwargs
    )

    return Source(train_loader, test_loader)
