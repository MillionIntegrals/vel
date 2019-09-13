import typing
import torch.utils.data as data

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
