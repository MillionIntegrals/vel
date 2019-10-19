import typing
import torch.utils.data as data

from vel.api import Source

from vel.data.dataflow import DataFlow


class DatasetLoader:
    """ Loads data from a data source to serve it to the model """

    def __init__(self, source: Source, batch_size: int, num_workers: int,
                 transformations: typing.Optional[list] = None, pin_memory=False):
        self._source = source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformations = transformations
        self.pin_memory = pin_memory

        if transformations is not None:
            self.transformed_source = DataFlow.transform(self._source, transformations)
        else:
            self.transformed_source = source

        self.train_loader = data.DataLoader(
            self.transformed_source.train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=True
        )

        self.val_loader = data.DataLoader(
            self.transformed_source.validation, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory
        )

        if self.transformed_source.test is not None:
            self.test_loader = data.DataLoader(
                self.transformed_source.test, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        else:
            self.test_loader = None

        self._loaders = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader
        }

        self._loader_sizes = {
            'train': len(self.train_loader),
            'val': len(self.val_loader),
            'test': 0 if self.test_loader is None else len(self.test_loader)
        }

    def __getitem__(self, item):
        return self._loaders[item]

    @property
    def source(self):
        """ Return the source for this loader """
        return self.transformed_source

    @property
    def loader(self):
        """ Get a dict of loaders """
        return self._loaders

    @property
    def size(self):
        """ Get a dict of sizes of each loader """
        return self._loader_sizes


def create(source: Source, batch_size: int, num_workers: int = 0, transformations: typing.Optional[list] = None,
           pin_memory=False):
    """ Vel factory function """
    return DatasetLoader(
        source=source,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        transformations=transformations
    )
