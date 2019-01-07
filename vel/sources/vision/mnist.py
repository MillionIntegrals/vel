from torchvision import datasets


from vel.api import TrainingData

from vel.augmentations.normalize import Normalize
from vel.augmentations.to_tensor import ToTensor
from vel.augmentations.to_array import ToArray


def create(model_config, batch_size, normalize=True, num_workers=0, augmentations=None):
    """ Create a MNIST dataset, normalized """
    path = model_config.data_dir('mnist')

    train_dataset = datasets.MNIST(path, train=True, download=True)
    test_dataset = datasets.MNIST(path, train=False, download=True)

    augmentations = [ToArray()] + (augmentations if augmentations is not None else [])

    if normalize:
        train_data = train_dataset.train_data
        mean_value = (train_data.double() / 255).mean().item()
        std_value = (train_data.double() / 255).std().item()

        augmentations.append(Normalize(mean=mean_value, std=std_value, tags=['train', 'val']))

    augmentations.append(ToTensor())

    return TrainingData(
        train_dataset,
        test_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        augmentations=augmentations
    )
