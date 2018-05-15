import yaml

from .provider import Provider
from ..exceptions import WbInitializationException


class ModelConfig:
    """ Read from YAML configuration of a model, specifying all details of the run """

    def __init__(self, filename, run_number, project_config):
        self.filename = filename

        with open(self.filename, 'r') as f:
            self.contents = yaml.safe_load(f)

        # Options that should exist for every config
        try:
            self.name = self.contents['name']
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

    def run_command(self, command_name):
        """ Instantiate model class """
        command_descriptor = self.provider.instantiate_from_data(self.command_descriptor[command_name])
        return self.provider.resolve_and_call(command_descriptor.run)

    def checkpoint_dir(self) -> str:
        """ Return checkpoint directory for this model """
        return self.project_config.project_output_dir('checkpoints', self.run_name)

    def checkpoint_filename(self, epoch) -> str:
        """ Return checkpoint filename for this model """
        return self.project_config.project_output_dir(
            'checkpoints', self.run_name, 'checkpoint_{:08}.npy'.format(epoch)
        )

    @property
    def run_name(self) -> str:
        """ Return name of the run """
        return "{}/{}".format(self.name, self.run_number)

    def banner(self, command_name) -> None:
        """ Print a banner for running the system """
        print("=" * 80)
        print("Running model {}, run {} -- command {}".format(self.name, self.run_number, command_name))
        print("=" * 80)
