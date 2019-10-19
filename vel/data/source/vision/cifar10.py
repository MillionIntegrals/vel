from torchvision import datasets

from vel.api import Source


def create(model_config):
    """
    Create a CIFAR10 dataset, normalized.
    Augmentations are the same as in the literature benchmarking CIFAR performance.
    """
    path = model_config.data_dir('cifar10')

    train_dataset = datasets.CIFAR10(path, train=True, download=True)
    test_dataset = datasets.CIFAR10(path, train=False, download=True)

    train_data = train_dataset.data
    mean_value = (train_data / 255).mean(axis=(0, 1, 2))
    std_value = (train_data / 255).std(axis=(0, 1, 2))

    return Source(
        train=train_dataset,
        validation=test_dataset,
        metadata={
            'train_mean': mean_value,
            'train_std': std_value
        }
    )

    # augmentations = [ToArray()] + (augmentations if augmentations is not None else [])

    # if normalize:
    #
    #     augmentations.append(Normalize(mean=mean_value, std=std_value, tags=['train', 'val']))
    #
    # augmentations.append(ToTensor())
    #
    # return SupervisedTrainingData(
    #     train_dataset,
    #     test_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     augmentations=augmentations
    # )
