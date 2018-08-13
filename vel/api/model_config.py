import torch
import yaml
import datetime as dtm

from vel.internals.provider import Provider
from vel.internals.project_config import ProjectConfig
from ..exceptions import VelInitializationException


class ModelConfig:
    """
    Read from YAML configuration of a model, specifying all details of the run.
    Is a frontend for the provider, resolving all dependency-injection requests.
    """

    def __init__(self, filename: str, run_number: int, project_config: ProjectConfig, reset=False, **kwargs):
        self.filename = filename
        self.device = kwargs.get('device', 'cuda')
        self.reset = reset

        with open(self.filename, 'r') as f:
            self.contents = yaml.safe_load(f)

        # Options that should exist for every config
        try:
            self._model_name = self.contents['name']
        except KeyError:
            raise VelInitializationException("Model configuration must have a 'name' key")

        self.run_number = run_number
        self.project_config = project_config

        self.command_descriptor = self.contents['commands']

        # This one is special and needs to get removed
        del self.contents['commands']

        self.provider = Provider(self._prepare_environment(), {
            'model_config': self,
            'project_config': self.project_config
        })

    def _prepare_environment(self) -> dict:
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

    def project_dir(self, *args) -> str:
        """ Return data directory for given dataset """
        return self.project_config.project_toplevel_dir(*args)

    def openai_dir(self) -> str:
        """ Return directory for openai output files for this model """
        return self.project_config.project_output_dir('openai', self.run_name)

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
    # PROVIDER API
    def provide(self, name):
        """ Return a dependency-injected instance """
        return self.provider.instantiate_by_name(name)

    ####################################################################################################################
    # BANNERS - Maybe shouldn't be here, but they are for now
    def banner(self, command_name) -> None:
        """ Print a banner for running the system """
        device = torch.device(self.device)
        print("=" * 80)
        print(f"Pytorch version: {torch.__version__} cuda version {torch.version.cuda} cudnn version {torch.backends.cudnn.version()}")
        print("Running model {}, run {} -- command {} -- device {}".format(self._model_name, self.run_number, command_name, self.device))
        if device.type == 'cuda':
            device_idx = 0 if device.index is None else device.index
            print(f"CUDA Device name {torch.cuda.get_device_name(device_idx)}")
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
        return f"<ModelConfig at {self.filename}>"
