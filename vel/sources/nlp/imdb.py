import os
import glob
import io
import pickle

import torchtext.datasets.imdb as imdb
import torchtext.data as data


from vel.api import TextData


class IMDBCached(imdb.IMDB):
    """ Cached version of the IMDB dataset (to save time on tokenization) """

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        cache_file = os.path.join(path, 'examples_cache.pk')

        fields = [('text', text_field), ('label', label_field)]

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                examples = pickle.load(fp)
        else:
            examples = []

            for label in ['pos', 'neg']:
                for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                    with io.open(fname, 'r', encoding="utf-8") as f:
                        text = f.readline()
                    examples.append(data.Example.fromlist([text, label], fields))

            with open(cache_file, 'wb') as fp:
                pickle.dump(examples, file=fp)

        data.Dataset.__init__(self, examples, fields, **kwargs)


def create(model_config, batch_size, vectors=None):
    """ Create an IMDB dataset """
    path = model_config.data_dir('imdb')

    text_field = data.Field(lower=True, tokenize='spacy', batch_first=True)
    label_field = data.LabelField(is_target=True)

    train_source, test_source = IMDBCached.splits(
        root=path,
        text_field=text_field,
        label_field=label_field
    )

    text_field.build_vocab(train_source, max_size=25_000, vectors=vectors)
    label_field.build_vocab(train_source)

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_source, test_source),
        batch_size=batch_size,
        device=model_config.torch_device(),
        shuffle=True
    )

    return TextData(
        train_source, test_source, train_iterator, test_iterator, text_field, label_field
    )
