import collections
import torch
import torch.utils.data as data
import typing

from vel.api import Source, Transformation


def pre_map(datapoint):
    """ Map datapoint from a list into the dictionary """
    if isinstance(datapoint, (list, tuple)):
        return dict(zip("xyzw", datapoint))

    if 'x' in datapoint:
        datapoint['size'] = datapoint['x'].shape[0]

    return datapoint


class DataFlow(data.Dataset):
    """ A dataset wrapping underlying data source with transformations """

    @staticmethod
    def transform(source: Source, transformations: typing.List[Transformation]) -> Source:
        """ Transform supplied source with a list of given transformations """
        # Initialize transformations from source
        for t in transformations:
            t.initialize(source)

        return Source(
            train=DataFlow(source.train, transformations, 'train'),
            validation=DataFlow(source.validation, transformations, 'val'),
            test=None if source.test is None else DataFlow(source.test, transformations, 'test')
        )

    def __init__(self, dataset, transformations, tag):
        self.dataset = dataset
        self.tag = tag

        if transformations is None:
            self.transformations = []
        else:
            self.transformations = [t for t in transformations if tag in t.tags]

    def get_raw(self, index):
        """ Get raw data point """
        return pre_map(self.dataset[index])

    def get_batch(self, batch_idx, batch_size):
        """
        Simple method to get a batch of data, mainly for interactive purposes.
        For training, a DataLoader should be used.
        """

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self))

        buffer = collections.defaultdict(list)

        for i in range(start_idx, end_idx):
            datapoint = self[i]

            for k, v in datapoint.items():
                buffer[k].append(v)

        return {
            k: torch.stack(v, dim=0) for k, v in buffer.items()
        }

    def num_batches(self, batch_size):
        """ Number of batches of given batch size """
        length = len(self)
        return (length + (batch_size - 1)) // batch_size

    def __getitem__(self, index):
        """ Get data point from the dataset """
        datapoint = self.get_raw(index)

        for t in self.transformations:
            datapoint = t(datapoint)

        return datapoint

    def denormalize(self, datapoint):
        """ Perform a reverse normalization (for viewing) """
        for t in self.transformations[::-1]:
            datapoint = t.denormalize(datapoint)

        return datapoint

    def denormalize_item(self, datapoint_item, coordinate):
        """ Perform a reverse normalization of a single item (for viewing) """
        for t in self.transformations[::-1]:
            datapoint_item = t.denormalize_item(datapoint_item, coordinate)

        return datapoint_item

    def __len__(self):
        """ Length of the dataset """
        return len(self.dataset)
