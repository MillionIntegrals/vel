import certifi
import numpy as np
import os
import pathlib
import urllib3

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
        self.padded_sequence_next = padded_sequence[1:].reshape(self.num_batches * self.batch_size, self.sequence_length)

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

            return input_data.to(torch.long), target_data.to(torch.long)


class TextLoader:
    """ Loader of sequential text data """
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


class TextUrlSource(Source):
    """ Download text from source and model it character by character """
    def __init__(self, url, absolute_data_path, sequence_length, batch_size, train_val_split=0.8):
        super().__init__()

        self.url = url
        self.data_path = absolute_data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_val_split = train_val_split

        self.text_path = os.path.join(self.data_path, 'text.txt')
        self.processed_path = os.path.join(self.data_path, 'processed.data')

        self.data_dict = self.download()

        content_encoded = self.data_dict['content_encoded']
        alphabet_size = len(self.data_dict['alphabet'])

        split_idx = int(len(content_encoded) * train_val_split)

        self._train_loader = TextLoader(
            sequence=content_encoded[:split_idx],
            sequence_length=sequence_length,
            batch_size=batch_size,
            alphabet_size=alphabet_size,
        )

        self._val_loader = TextLoader(
            sequence=content_encoded[split_idx:],
            sequence_length=sequence_length,
            batch_size=batch_size,
            alphabet_size=alphabet_size,
        )

    def encode_character(self, char):
        return self.data_dict['character_to_index'][char]

    def decode_character(self, index):
        return self.data_dict['index_to_character'][index]

    def train_loader(self):
        """ PyTorch loader of training data """
        return self._train_loader

    def val_loader(self):
        """ PyTorch loader of validation data """
        return self._val_loader

    def train_dataset(self):
        """ Return the training dataset """
        return None

    def val_dataset(self):
        """ Return the validation dataset """
        return None

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        return len(self._train_loader)

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        return len(self._val_loader)

    def download(self):
        """ Make sure data file is downloaded and stored properly """
        if not os.path.exists(self.data_path):
            # Create if it doesn't exist
            pathlib.Path(self.data_path).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.text_path):
            http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

            with open(self.text_path, 'wt') as fp:
                request = http.request('GET', self.url)
                content = request.data.decode('utf8')
                fp.write(content)

        if not os.path.exists(self.processed_path):
            with open(self.text_path, 'rt') as fp:
                content = fp.read()

            alphabet = sorted(set(content))

            index_to_character = {idx: c for idx, c in enumerate(alphabet, 1)}
            character_to_index = {c: idx for idx, c in enumerate(alphabet, 1)}

            content_encoded = np.array([character_to_index[c] for c in content], dtype=np.uint8)

            data_dict = {
                'alphabet': alphabet,
                'index_to_character': index_to_character,
                'character_to_index': character_to_index,
                'content_encoded': content_encoded
            }

            with open(self.processed_path, 'wb') as fp:
                torch.save(data_dict, fp)
        else:
            with open(self.processed_path, 'rb') as fp:
                data_dict = torch.load(fp)

        return data_dict


def create(model_config, url, local_dir, sequence_length=64, batch_size=64, train_val_split=0.8):
    """ Vel factory function """
    if not os.path.isabs(local_dir):
        local_dir = model_config.project_data_dir(local_dir)

    return TextUrlSource(
        url,
        absolute_data_path=local_dir,
        sequence_length=sequence_length,
        batch_size=batch_size,
        train_val_split=train_val_split,
    )
