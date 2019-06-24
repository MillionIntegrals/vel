from torchvision import datasets

from vel.api import Source



def create(model_config):
    """ Create a MNIST dataset, normalized """
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

# from vel.api import SupervisedTrainingData
#
# from vel.augmentations.normalize import Normalize
# from vel.augmentations.to_tensor import ToTensor
# from vel.augmentations.to_array import ToArray
# from vel.augmentations.unsupervised import Unsupervised

    # augmentations = [ToArray()] + (augmentations if augmentations is not None else [])
    #
    # if normalize:
    #
    #     augmentations.append(Normalize(mean=mean_value, std=std_value, tags=['train', 'val']))
    #
    # augmentations.append(ToTensor())
    #
    # if unsupervised:
    #     augmentations.append(Unsupervised())
    #
    # return SupervisedTrainingData(
    #     train_dataset,
    #     test_dataset,
    #     num_workers=num_workers,
    #     batch_size=batch_size,
    #     augmentations=augmentations
    # )
