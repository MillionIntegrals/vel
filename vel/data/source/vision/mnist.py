from torchvision import datasets

from vel.api import Source


def create(model_config):
    """ Create a MNIST dataset """
    path = model_config.data_dir('mnist')

    train_dataset = datasets.MNIST(path, train=True, download=True)
    test_dataset = datasets.MNIST(path, train=False, download=True)

    train_data = train_dataset.data
    mean_value = (train_data.double() / 255).mean().item()
    std_value = (train_data.double() / 255).std().item()

    return Source(
        train=train_dataset,
        validation=test_dataset,
        metadata={
            'train_mean': mean_value,
            'train_std': std_value
        }
    )
