import torchtext.data as data

from vel.util.dataloader import IteratorDictWrapper
from vel.api import LanguageSource, ModelConfig


class BucketLoader:
    """ Loads sequence data from a source and batches together examples of similar length """

    def __init__(self, model_config: ModelConfig, source: LanguageSource, batch_size: int):
        self._source = source
        self.batch_size = batch_size

        if self._source.test is None:
            self.train_loader, self.val_loader = data.BucketIterator.splits(
                (self._source.train, self._source.validation),
                batch_size=batch_size,
                device=model_config.torch_device(),
                shuffle=True
            )
            self.test_loader = None
        else:
            self.train_loader, self.val_loader, self.test_loader = data.BucketIterator.splits(
                (self._source.train, self._source.validation, self._source.test),
                batch_size=batch_size,
                device=model_config.torch_device(),
                shuffle=True
            )

        self.train_loader = IteratorDictWrapper(self.train_loader, self._source.mapping)
        self.val_loader = IteratorDictWrapper(self.val_loader, self._source.mapping)

        if self.test_loader:
            self.test_loader = IteratorDictWrapper(self.test_loader, self._source.mapping)

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
        return self._source

    @property
    def loader(self):
        """ Get a dict of loaders """
        return self._loaders

    @property
    def size(self):
        """ Get a dict of sizes of each loader """
        return self._loader_sizes

    @property
    def alphabet_size(self):
        """ Size of the text alphabet """
        return self._source.metadata.get('alphabet_size', 0)


def create(model_config: ModelConfig, source: LanguageSource, batch_size: int):
    """ Vel factory function """
    return BucketLoader(
        model_config=model_config,
        source=source,
        batch_size=batch_size,
    )
