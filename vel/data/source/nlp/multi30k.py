import io
import os.path
import pickle
import re
import spacy

import torchtext.data as data
import torchtext.datasets as ds

from vel.api import SupervisedTextData


class Multi30kCached(ds.Multi30k):
    """ Cached version of the Multi30K dataset, to save time on tokenization every time """

    def __init__(self, path, exts, fields, **kwargs):
        # Each one is a
        if os.path.isdir(path):
            cache_file = os.path.join(path, '_cache.pk')
        else:
            cache_file = path + '_cache.pk'

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                examples = pickle.load(fp)
        else:
            src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

            examples = []

            with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                    io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
                for src_line, trg_line in zip(src_file, trg_file):
                    src_line, trg_line = src_line.strip(), trg_line.strip()
                    if src_line != '' and trg_line != '':
                        examples.append(data.Example.fromlist(
                            [src_line, trg_line], fields))

            with open(cache_file, 'wb') as fp:
                pickle.dump(examples, file=fp)

        data.Dataset.__init__(self, examples, fields, **kwargs)


def create(model_config, batch_size, data_dir='wmt14'):
    """ Create an Multi30k dataset. English-German """
    path = model_config.data_dir(data_dir)

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    en_field = data.Field(
        lower=True, tokenize=tokenize_en, batch_first=True, init_token='<sos>', eos_token='<eos>'
    )

    de_field = data.Field(
        lower=True, tokenize=tokenize_de, batch_first=True, init_token='<sos>', eos_token='<eos>'
    )

    train_source, val_source, test_source = Multi30kCached.splits(
        root=path,
        exts=('.en', '.de'),
        fields=(en_field, de_field)
    )

    en_field.build_vocab(train_source.src, min_freq=2)
    de_field.build_vocab(train_source.tgt, max_size=17_000)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_source, val_source, test_source),
        batch_size=batch_size,
        repeat=False
    )

    return SupervisedTextData(
        train_source, val_source, train_iter, val_iter, en_field, de_field
    )
