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

#     @property
#     def train_loader(self):
#         """ PyTorch loader of training data """
#         raise NotImplementedError
#
#     @property
#     def val_loader(self):
#         """ PyTorch loader of validation data """
#         raise NotImplementedError
#
#     @property
#     def train_dataset(self):
#         """ Return the training dataset """
#         raise NotImplementedError
#
#     @property
#     def val_dataset(self):
#         """ Return the validation dataset """
#         raise NotImplementedError
#
#     @property
#     def train_iterations_per_epoch(self):
#         """ Return number of iterations per epoch """
#         raise NotImplementedError
#
#     @property
#     def val_iterations_per_epoch(self):
#         """ Return number of iterations per epoch - validation """
#         raise NotImplementedError
#
#
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
#
#
# class SupervisedTrainingData(Source):
#     """ Most common source of data combining a basic datasource and sampler """
#     def __init__(self, train_source, val_source, num_workers, batch_size, augmentations=None):
#
#         super().__init__()
#
#         self.train_source = train_source
#         self.val_source = val_source
#
#         self.num_workers = num_workers
#         self.batch_size = batch_size
#
#         self.augmentations = augmentations
#
#         # Derived values
#         self.train_ds = DataFlow(self.train_source, augmentations, tag='train')
#         self.val_ds = DataFlow(self.val_source, augmentations, tag='val')
#
#         self._train_loader = data.DataLoader(
#             self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
#         )
#
#         self._val_loader = data.DataLoader(
#             self.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )
#
#     @property
#     def train_loader(self):
#         """ PyTorch loader of training data """
#         return self._train_loader
#
#     @property
#     def val_loader(self):
#         """ PyTorch loader of validation data """
#         return self._val_loader
#
#     @property
#     def train_dataset(self):
#         """ Return the training dataset """
#         return self.train_ds
#
#     @property
#     def val_dataset(self):
#         """ Return the validation dataset """
#         return self.val_ds
#
#     @property
#     def train_iterations_per_epoch(self):
#         """ Return number of iterations per epoch """
#         return len(self._train_loader)
#
#     @property
#     def val_iterations_per_epoch(self):
#         """ Return number of iterations per epoch - validation """
#         return len(self._val_loader)
