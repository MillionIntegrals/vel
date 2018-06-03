import yaml
import datetime as dtm

from waterboy.internals.provider import Provider
from ..exceptions import WbInitializationException


class ModelConfig:
    """
    Read from YAML configuration of a model, specifying all details of the run.
    Is a frontend for the provider, resolving all dependency-injection requests.
    """

    def __init__(self, filename, run_number, project_config, **kwargs):
        self.filename = filename
        self.device = kwargs.get('device', 'cuda')

        with open(self.filename, 'r') as f:
            self.contents = yaml.safe_load(f)

        # Options that should exist for every config
        try:
            self._model_name = self.contents['name']
        except KeyError:
            raise WbInitializationException("Model configuration must have a 'name' key")

        self.run_number = run_number
        self.project_config = project_config

        self.command_descriptor = self.contents['commands']

        # This one is special and needs to get removed
        del self.contents['commands']

        self.provider = Provider(self._prepare_environment(), {
            'model_config': self,
            'project_config': self.project_config
        })

    def _prepare_environment(self):
        """ Return full environment for dependency injection """
        return {
            **self.project_config.contents,
            **self.contents,
            'run_number': self.run_number
        }

    ####################################################################################################################
    # COMMAND UTILITIES
    def get_command(self, command_name):
        """ Return object for given command """
        return self.provider.instantiate_from_data(self.command_descriptor[command_name])

    def run_command(self, command_name, varargs):
        """ Instantiate model class """
        command_descriptor = self.get_command(command_name)
        return command_descriptor.run(*varargs)

    ####################################################################################################################
    # MODEL DIRECTORIES
    def checkpoint_dir(self, *args) -> str:
        """ Return checkpoint directory for this model """
        return self.project_config.project_output_dir('checkpoints', self.run_name, *args)

    def data_dir(self, *args) -> str:
        """ Return data directory for given dataset """
        return self.project_config.project_data_dir(*args)

    def output_dir(self, *args) -> str:
        """ Return data directory for given dataset """
        return self.project_config.project_output_dir(*args)

    ####################################################################################################################
    # NAME UTILITIES
    @property
    def run_name(self) -> str:
        """ Return name of the run """
        return "{}/{}".format(self._model_name, self.run_number)

    @property
    def name(self) -> str:
        """ Return name of the run """
        return self._model_name

    ####################################################################################################################
    # BANNERS - Maybe shouldn't be here, but they are for now
    def banner(self, command_name) -> None:
        """ Print a banner for running the system """
        print("=" * 80)
        print("Running model {}, run {} -- command {} -- device {}".format(self._model_name, self.run_number, command_name, self.device))
        print(dtm.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"))
        print("=" * 80)

    def quit_banner(self) -> None:
        """ Print a banner for running the system """
        print("=" * 80)
        print("Done.")
        print(dtm.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"))
        print("=" * 80)

    ####################################################################################################################
    # Small UI utils
    def __repr__(self):
        return f"<ModelConfig at {self.filename}"
