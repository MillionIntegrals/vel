from torchvision import datasets

from vel.api import Source


def create(model_config):
    """ Create an Omniglot dataset """
    path = model_config.data_dir('omniglot')

    train_dataset = datasets.Omniglot(path, background=True, download=True)
    test_dataset = datasets.Omniglot(path, background=False, download=True)

    return Source(
        train=train_dataset,
        validation=test_dataset,
    )
