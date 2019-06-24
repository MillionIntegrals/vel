import typing
import torch.utils.data as data

from vel.api import Source, Transformation


def pre_map(datapoint):
    """ Map datapoint from a list into the dictionary """
    if isinstance(datapoint, (list, tuple)):
        return dict(zip("xyzw", datapoint))
    return datapoint


class DataFlow(data.Dataset):
    """ A dataset wrapping underlying data source with transformations """

    @staticmethod
    def transform(source: Source, transformations: typing.List[Transformation]) -> Source:
        """ Transform supplied source with a list of given transformations """
        return Source(
            train=DataFlow(source.train, transformations, 'train'),
            validation=DataFlow(source.validation, transformations, 'val'),
            test=None if source.test is None else DataFlow(source.test, transformations, 'test')
        )

    def __init__(self, dataset, transformations, tag):
        self.dataset = dataset

        if transformations is None:
            self.transformations = []
        else:
            self.transformations = [t for t in transformations if tag in t.tags]

        self.tag = tag

    def get_raw(self, index):
        return pre_map(self.dataset[index])

    def __getitem__(self, index):
        datapoint = self.get_raw(index)

        for t in self.transformations:
            datapoint = t(datapoint)

        return datapoint

    def denormalize(self, datapoint):
        """ Perform a reverse normalization (for viewing) """
        for t in self.transformations[::-1]:
            datapoint = t.denormalize(datapoint)

        return datapoint

    def __len__(self):
        return len(self.dataset)
