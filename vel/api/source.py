import typing

import torch.utils.data as data


class Source:
    """
    Single simple container for train/validation/test datasets.

    PyTorch datasets by default support only __len__ and __getitem__ operations
    """

    def __init__(self, train: data.Dataset, validation: data.Dataset,
                 test: typing.Optional[data.Dataset] = None, metadata: typing.Optional[dict] = None):
        self.train = train
        self.validation = validation
        self.test = test

        self.metadata = {} if metadata is None else metadata
