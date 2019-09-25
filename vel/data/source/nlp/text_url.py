import certifi
import numpy as np
import os
import pathlib
import urllib3

import torch

from vel.api import Source


class TextUrlSource(Source):
    """ Download text from source and model it character by character """

    def __init__(self, url, absolute_data_path, train_val_split=0.8):
        self.url = url
        self.data_path = absolute_data_path
        self.train_val_split = train_val_split

        self.text_path = os.path.join(self.data_path, 'text.txt')
        self.processed_path = os.path.join(self.data_path, 'processed.data')

        self.data_dict = self.download()

        content_encoded = self.data_dict['content_encoded']

        split_idx = int(len(content_encoded) * train_val_split)

        super().__init__(
            train=content_encoded[:split_idx],
            validation=content_encoded[split_idx:],
            metadata={
                'alphabet': self.data_dict['alphabet'],
                'character_to_index': self.data_dict['character_to_index'],
                'index_to_character': self.data_dict['index_to_character']
            }
        )

    def download(self) -> dict:
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


def create(model_config, url, local_dir, train_val_split=0.8):
    """ Vel factory function """
    if not os.path.isabs(local_dir):
        local_dir = model_config.data_dir(local_dir)

    return TextUrlSource(
        url,
        absolute_data_path=local_dir,
        train_val_split=train_val_split,
)
