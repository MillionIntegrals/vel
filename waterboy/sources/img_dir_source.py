import os.path

import torchvision.datasets as ds
import torch.utils.data as data

from waterboy.api.base import Source
from waterboy.api.data import DataFlow


class ImageDirSource(ds.ImageFolder):
    pass


def create(model_config, path, num_workers, batch_size, augmentations=None):
    """ Create an ImageDirSource with supplied arguments """
    if not os.path.isabs(path):
        path = model_config.project_dir(path)

    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'valid')

    train_ds = ImageDirSource(train_path)
    val_ds = ImageDirSource(valid_path)

    train_df = DataFlow(train_ds, augmentations, tag='train')
    val_df = DataFlow(val_ds, augmentations, tag='val')

    train_loader = data.DataLoader(
        train_df, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = data.DataLoader(
        val_df, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return Source(train_loader, test_loader, batch_size=batch_size)

