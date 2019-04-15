import os.path
import zipfile

import torchvision.datasets as ds
import torchvision.datasets.utils as ds_util

from vel.api import SupervisedTrainingData


class ImageDirSource(ds.ImageFolder):
    """ Source where images are grouped by class in folders """
    pass


def create(model_config, path, num_workers, batch_size, augmentations=None, tta=None, url=None,
           extract_parent=False):
    """ Create an ImageDirSource with supplied arguments """
    if not os.path.isabs(path):
        path = model_config.project_top_dir(path)

    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'valid')

    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        filename = url.rpartition('/')[2]
        ds_util.download_url(url, root=path, filename=filename)

        full_archive_path = os.path.join(path, filename)

        # Unpack zip archive
        if full_archive_path.endswith(".zip"):
            zip_ref = zipfile.ZipFile(full_archive_path, 'r')

            if extract_parent:
                zip_ref.extractall(os.path.dirname(path))
            else:
                zip_ref.extractall(path)

            zip_ref.close()

            os.remove(full_archive_path)

    train_ds = ImageDirSource(train_path)
    val_ds = ImageDirSource(valid_path)

    return SupervisedTrainingData(
        train_ds,
        val_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        augmentations=augmentations,
        # test_time_augmentation=tta
    )
