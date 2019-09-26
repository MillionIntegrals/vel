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


class LanguageSource(Source):
    """ Special source for language datasets that handles dictionaries/encodings """

    def __init__(self, train: data.Dataset, validation: data.Dataset,
                 fields: dict, mapping: dict,
                 test: typing.Optional[data.Dataset] = None, metadata: typing.Optional[dict] = None):
        super().__init__(train, validation, test, metadata)

        self.fields = fields
        self.mapping = mapping


# class SupervisedTextData(Source):
#     """ An NLP torchtext data source """
#     def __init__(self, train_source, val_source, train_iterator, val_iterator, data_field, target_field):
#         super().__init__()
#
#         self.train_source = train_source
#         self.val_source = val_source
#         self.train_iterator = train_iterator
#         self.val_iterator = val_iterator
#         self.data_field = data_field
#         self.target_field = target_field
#
#     @property
#     def train_loader(self):
#         """ PyTorch loader of training data """
#         return self.train_iterator
#
#     @property
#     def val_loader(self):
#         """ PyTorch loader of validation data """
#         return self.val_iterator
#
#     @property
#     def train_dataset(self):
#         """ Return the training dataset """
#         return self.train_source
#
#     @property
#     def val_dataset(self):
#         """ Return the validation dataset """
#         return self.val_source
#
#     @property
#     def train_iterations_per_epoch(self):
#         """ Return number of iterations per epoch """
#         return len(self.train_iterator)
#
#     @property
#     def val_iterations_per_epoch(self):
#         """ Return number of iterations per epoch - validation """
#         return len(self.val_iterator)
