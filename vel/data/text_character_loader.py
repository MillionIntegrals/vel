import numpy as np
import torch

from vel.api import Source


class TextIterator:
    """ Iterator over a text dataset """
    def __init__(self, padded_sequence, sequence_length, batch_size, alphabet_size, num_batches):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.alphabet_size = alphabet_size

        self.padded_sequence = padded_sequence[:-1].reshape(self.num_batches * self.batch_size, self.sequence_length)
        self.padded_sequence_next = padded_sequence[1:].reshape(
            self.num_batches * self.batch_size, self.sequence_length
        )

        self.sequence_indices = np.arange(self.num_batches * self.batch_size)

        np.random.shuffle(self.sequence_indices)

        self.sequence_indices = self.sequence_indices.reshape(self.num_batches, self.batch_size)

        self.batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx == self.num_batches:
            raise StopIteration
        else:
            input_data = torch.from_numpy(self.padded_sequence[self.sequence_indices[self.batch_idx]])
            target_data = torch.from_numpy(self.padded_sequence_next[self.sequence_indices[self.batch_idx]])

            self.batch_idx += 1

            return {'x': input_data.to(torch.long), 'y': target_data.to(torch.long)}


class TextLoader:
    """ Creates iterators over a sequential block of text """
    def __init__(self, sequence, sequence_length, batch_size, alphabet_size):
        self.sequence = sequence
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.alphabet_size = alphabet_size

        # 1 is for the last element as the target needs to be shifted by 1
        residual_length = (len(self.sequence) - self.sequence_length - 1)
        full_size = self.sequence_length * self.batch_size

        rest = residual_length % full_size
        self.num_batches = residual_length // full_size

        if rest > 0:
            self.sequence = np.pad(self.sequence, (0, full_size - rest), mode='constant')
            self.num_batches += 1

    def __iter__(self):
        initial_offset = np.random.randint(self.sequence_length)
        relevant_subsequence = self.sequence[
            # 1 is for the last element as the target needs to be shifted by 1
            initial_offset:self.num_batches * self.sequence_length * self.batch_size + initial_offset + 1
        ]

        return TextIterator(
            relevant_subsequence, self.sequence_length, self.batch_size,
            alphabet_size=self.alphabet_size,
            num_batches=self.num_batches
        )

    def __len__(self):
        """ Number of batches in this loader """
        return self.num_batches


class TextCharacterLoader:
    """ Loader for the text character data source """

    def __init__(self, source, sequence_length: int, batch_size: int):
        self.source = source
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.alphabet = self.source.metadata['alphabet']

        self.train_loader = TextLoader(self.source.train, self.sequence_length, self.batch_size, len(self.alphabet))
        self.val_loader = TextLoader(self.source.validation, self.sequence_length, self.batch_size, len(self.alphabet))

        if self.source.test is None:
            self.test_loader = None
        else:
            self.test_loader = TextLoader(self.source.test, self.sequence_length, self.batch_size, len(self.alphabet))

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
    def alphabet_size(self):
        """ Size of the text alphabet """
        return len(self.alphabet)

    @property
    def loader(self):
        """ Get a dict of loaders """
        return self._loaders

    @property
    def size(self):
        """ Get a dict of sizes of each loader """
        return self._loader_sizes


def create(source: Source, sequence_length: int = 64, batch_size: int = 64):
    """ Vel factory function """
    return TextCharacterLoader(
        source=source,
        sequence_length=sequence_length,
        batch_size=batch_size
    )
