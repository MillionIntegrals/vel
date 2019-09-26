import os
import os.path
import glob
import json

from vel.api import ModelConfig


class ListCommand:
    """ List trained models for given config and their basic metadata """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def run(self):
        meta_dir = self.model_config.output_dir('meta', self.model_config.name)
        meta_paths = os.path.join(meta_dir, '*', 'meta.json')

        for path in sorted(glob.glob(meta_paths)):
            with open(path, 'rt') as fp:
                meta = json.load(fp)

            print("-" * 80)
            print("Run name: {}".format(meta['run_name']))
            print("Tag: {}".format(meta['tag']))
            print("Created: {}".format(meta['created']))


def create(model_config):
    """ Vel factory function """
    return ListCommand(
        model_config=model_config,
    )
