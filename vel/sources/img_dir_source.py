import os.path

import torchvision.datasets as ds

from vel.api import TrainingData


class ImageDirSource(ds.ImageFolder):
    pass


def create(model_config, path, num_workers, batch_size, augmentations=None, tta=None):
    """ Create an ImageDirSource with supplied arguments """
    if not os.path.isabs(path):
        path = model_config.project_top_dir(path)

    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'valid')

    train_ds = ImageDirSource(train_path)
    val_ds = ImageDirSource(valid_path)

    return TrainingData(
        train_ds,
        val_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        augmentations=augmentations,
        # test_time_augmentation=tta
    )
