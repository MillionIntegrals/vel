import torch
import torch.utils.data

from torchvision import datasets, transforms


from waterboy.api.source import Source


def create(batch_size, model_config, normalize=True, num_workers=0, augmentations=None):
    """
    Create a CIFAR10 dataset, normalized.
    Augmentations are the same as in the literature benchmarking CIFAR performance.
    """
    kwargs = {}
    augmentations = augmentations or []

    path = model_config.data_dir('cifar10')

    train_dataset = datasets.CIFAR10(path, train=True, download=True)
    test_dataset = datasets.CIFAR10(path, train=False, download=True)

    if normalize:
        train_data = train_dataset.train_data
        mean_value = (train_data / 255).mean(axis=(0, 1, 2))
        std_value = (train_data / 255).std(axis=(0, 1, 2))

        train_transform = transforms.Compose(augmentations + [
            transforms.ToTensor(),
            transforms.Normalize(mean_value, std_value)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_value, std_value)
        ])
    else:
        train_transform = transforms.Compose(augmentations + [transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, **kwargs
    )

    return Source(train_loader, test_loader)
