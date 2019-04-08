import datetime as dtm
import os.path

from vel.exceptions import VelInitializationException
from vel.internals.parser import Parser
from vel.internals.provider import Provider


class ModelConfig:
    """
    Read from YAML configuration of a model, specifying all details of the run.
    Is a frontend for the provider, resolving all dependency-injection requests.
    """

    PROJECT_FILE_NAME = '.velproject.yaml'

    @staticmethod
    def find_project_directory(start_path) -> str:
        """ Locate top-level project directory  """
        start_path = os.path.realpath(start_path)
        possible_name = os.path.join(start_path, ModelConfig.PROJECT_FILE_NAME)

        if os.path.exists(possible_name):
            return start_path
        else:
            up_path = os.path.realpath(os.path.join(start_path, '..'))
            if os.path.realpath(start_path) == up_path:
                raise RuntimeError(f"Couldn't find project file starting from {start_path}")
            else:
                return ModelConfig.find_project_directory(up_path)

    @classmethod
    def from_file(cls, filename: str, run_number: int, continue_training: bool = False, seed: int = None,
                  device: str = 'cuda', params=None):
        """ Create model config from file """
        with open(filename, 'r') as fp:
            model_config_contents = Parser.parse(fp)

        project_config_path = ModelConfig.find_project_directory(os.path.dirname(os.path.abspath(filename)))

        with open(os.path.join(project_config_path, cls.PROJECT_FILE_NAME), 'r') as fp:
            project_config_contents = Parser.parse(fp)

        aggregate_dictionary = {
            **project_config_contents,
            **model_config_contents
        }

        return ModelConfig(
            filename=filename,
            configuration=aggregate_dictionary,
            run_number=run_number,
            project_dir=project_config_path,
            continue_training=continue_training,
            seed=seed,
            device=device,
            parameters=params
        )

    @classmethod
    def from_memory(cls, model_data: dict, run_number: int, project_dir: str,
                    continue_training=False, seed: int = None, device: str = 'cuda', params=None):
        """ Create model config from supplied data """
        return ModelConfig(
            filename="[memory]",
            configuration=model_data,
            run_number=run_number,
            project_dir=project_dir,
            continue_training=continue_training,
            seed=seed,
            device=device,
            parameters=params
        )

    def __init__(self, filename: str, configuration: dict, run_number: int, project_dir: str,
                 continue_training=False, seed: int = None, device: str = 'cuda', parameters=None):
        self.filename = filename
        self.device = device
        self.continue_training = continue_training
        self.run_number = run_number
        self.seed = seed if seed is not None else (dtm.date.today().year + self.run_number)

        self.contents = configuration
        self.project_dir = os.path.normpath(project_dir)

        self.command_descriptors = self.contents.get('commands', [])

        # This one is special and needs to get removed
        if 'commands' in self.contents:
            del self.contents['commands']

        self.provider = Provider(self._prepare_environment(), {'model_config': self}, parameters=parameters)
        self._model_name = self.provider.get("name")

    def _prepare_environment(self) -> dict:
        """ Return full environment for dependency injection """
        return {**self.contents, 'run_number': self.run_number}

    def render_configuration(self) -> dict:
        """ Return a nice and picklable run configuration """
        return self.provider.render_configuration()

    ####################################################################################################################
    # COMMAND UTILITIES
    def get_command(self, command_name):
        """ Return object for given command """
        return self.provider.instantiate_from_data(self.command_descriptors[command_name])

    def run_command(self, command_name, varargs):
        """ Instantiate model class """
        command_descriptor = self.get_command(command_name)
        return command_descriptor.run(*varargs)

    ####################################################################################################################
    # MODEL DIRECTORIES
    def checkpoint_dir(self, *args) -> str:
        """ Return checkpoint directory for this model """
        return self.output_dir('checkpoints', self.run_name, *args)

    def data_dir(self, *args) -> str:
        """ Return data directory for given dataset """
        return self.project_data_dir(*args)

    def openai_dir(self) -> str:
        """ Return directory for openai output files for this model """
        return self.output_dir('openai', self.run_name)

    def project_data_dir(self, *args) -> str:
        """ Directory where to store data """
        return os.path.normpath(os.path.join(self.project_dir, 'data', *args))

    def output_dir(self, *args) -> str:
        """ Directory where to store output """
        return os.path.join(self.project_dir, 'output', *args)

    def project_top_dir(self, *args) -> str:
        """ Project top-level directory """
        return os.path.join(self.project_dir, *args)

    ####################################################################################################################
    # NAME UTILITIES
    @property
    def run_name(self) -> str:
        """ Return name of the run """
        return "{}/{}".format(self._model_name, self.run_number)

    @property
    def name(self) -> str:
        """ Return name of the model """
        return self._model_name

    ####################################################################################################################
    # MISC GETTERS
    def torch_device(self):
        """ Return torch device object """
        import torch
        return torch.device(self.device)

    ####################################################################################################################
    # PROVIDER API
    def provide(self, name):
        """ Return a dependency-injected instance """
        return self.provider.instantiate_by_name(name)

    def provide_with_default(self, name, default=None):
        """ Return a dependency-injected instance """
        return self.provider.instantiate_by_name_with_default(name, default_value=default)

    ####################################################################################################################
    # BANNERS - Maybe shouldn't be here, but they are for now
    def banner(self, command_name) -> None:
        """ Print a banner for running the system """
        import torch
        device = self.torch_device()

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
