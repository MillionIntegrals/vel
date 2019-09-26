import datetime as dtm
import json
import os.path
import pathlib
import typing

from vel.exception import VelInitializationException
from vel.internal.parser import Parser
from vel.internal.provider import Provider

from .info import TrainingInfo


class ModelConfig:
    """
    Read from YAML configuration of a model, specifying all details of the run.
    Is a frontend for the provider, resolving all dependency-injection requests.
    """

    PROJECT_FILE_NAME = '.velproject.yaml'
    META_FILE_NAME = 'meta.json'

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

    @staticmethod
    def from_project_directory(path) -> str:
        """ Locate given path relative to project directory """
        return os.path.join(ModelConfig.find_project_directory('.'), path)

    @classmethod
    def from_file(cls, filename: str, run_number: int = 1, resume_training: bool = False, seed: int = None,
                  device: str = 'cuda', parameters: typing.Optional[dict] = None, tag: typing.Optional[str] = None):
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
            resume_training=resume_training,
            seed=seed,
            device=device,
            parameters=parameters,
            tag=tag
        )

    @classmethod
    def script(cls, model_name: str = 'script', configuration: typing.Optional[dict] = None, run_number: int = 1,
               resume_training=False, seed: int = None, device: str = 'cuda',
               parameters: typing.Optional[dict] = None, tag: typing.Optional[str] = None):
        """ Create model config from supplied data """
        if configuration is None:
            configuration = {}

        configuration['name'] = model_name

        project_config_path = ModelConfig.find_project_directory(os.path.dirname(os.path.abspath(os.getcwd())))

        with open(os.path.join(project_config_path, cls.PROJECT_FILE_NAME), 'r') as fp:
            project_config_contents = Parser.parse(fp)

        aggregate_dictionary = {
            **project_config_contents,
            **configuration
        }

        return ModelConfig(
            filename="[script]",
            configuration=aggregate_dictionary,
            run_number=run_number,
            project_dir=project_config_path,
            resume_training=resume_training,
            seed=seed,
            device=device,
            parameters=parameters,
            tag=tag
        )

    def __init__(self, filename: str, configuration: dict, run_number: int, project_dir: str,
                 resume_training=False, seed: int = None, device: str = 'cuda',
                 parameters: typing.Optional[dict] = None, tag: typing.Optional[str] = None):
        self.filename = filename
        self.device = device
        self.resume_training = resume_training
        self.run_number = run_number
        self.seed = seed if seed is not None else (dtm.date.today().year + self.run_number)

        self.contents = configuration
        self.project_dir = os.path.normpath(project_dir)

        self.command_descriptors = {
            **self.contents.get('global_commands', {}),
            **self.contents.get('commands', {})
        }

        # This one is special and needs to get removed
        if 'commands' in self.contents:
            del self.contents['commands']

        if 'global_commands' in self.contents:
            del self.contents['global_commands']

        self.provider = Provider(self._prepare_environment(), {'model_config': self}, parameters=parameters)

        if self.provider.has_name('output_directory'):
            self.output_directory_name = self.provider.get("output_directory")
        else:
            self.output_directory_name = 'output'

        self._model_name = self.provider.get("name")

        if self.meta_exists():
            self._meta = self._load_meta()

            if resume_training:
                if (tag is not None) and (tag != self._meta['tag']):
                    raise VelInitializationException("Model tag mismatch")
                else:
                    self._tag = self._meta['tag']
            else:
                self._tag = tag
        else:
            self._tag = tag
            self._meta = None

    ####################################################################################################################
    # INTERNAL FUNCTIONS
    def _prepare_environment(self) -> dict:
        """ Return full environment for dependency injection """
        return {**self.contents, 'run_number': self.run_number}

    def _load_meta(self) -> dict:
        """ Load previously written metadata about the project """
        if not self.meta_exists():
            raise VelInitializationException("Previous run does not exist")

        with open(self.meta_dir(self.META_FILE_NAME), 'rt') as fp:
            return json.load(fp)

    def _create_meta(self) -> dict:
        """ Metadata for this model/config """
        return {
            'run_name': self.run_name,
            'tag': self.tag,
            'created': dtm.datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
            'config': self.render_configuration()
        }

    ####################################################################################################################
    # Metadata handling
    def meta_exists(self):
        """ If metadata file exists for this config """
        return os.path.exists(self.meta_dir(self.META_FILE_NAME))

    def enforce_meta(self):
        """ Make sure metadata exists for this config """
        if self._meta is None:
            raise VelInitializationException("Given model has not been initialized")

    def write_meta(self) -> None:
        """ Write metadata to a file """
        self._meta = self._create_meta()

        pathlib.Path(self.meta_dir()).mkdir(parents=True, exist_ok=True)

        with open(self.meta_dir(self.META_FILE_NAME), 'wt') as fp:
            return json.dump(self.meta, fp)

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
    def project_top_dir(self, *args) -> str:
        """ Project top-level directory """
        return os.path.join(self.project_dir, *args)

    def output_dir(self, *args) -> str:
        """ Directory where to store output """
        return os.path.join(self.project_dir, self.output_directory_name, *args)

    def meta_dir(self, *args) -> str:
        """ Return directory for openai output files for this model """
        return self.output_dir('meta', self.run_name, *args)

    def data_dir(self, *args) -> str:
        """ Directory where to store data """
        return os.path.normpath(os.path.join(self.project_dir, 'data', *args))

    def checkpoint_dir(self, *args) -> str:
        """ Return checkpoint directory for this model """
        return self.output_dir('checkpoints', self.run_name, *args)

    def openai_dir(self, *args) -> str:
        """ Return directory for openai output files for this model """
        return self.output_dir('openai', self.run_name, *args)

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

    @property
    def meta(self) -> dict:
        """ Return name of the model """
        self.enforce_meta()
        return self._meta

    @property
    def tag(self) -> typing.Optional[str]:
        """ Tag for this model/run number """
        return self._tag

    ####################################################################################################################
    # MISC GETTERS
    def torch_device(self):
        """ Return torch device object """
        import torch
        return torch.device(self.device)

    def render_configuration(self) -> dict:
        """ Return a nice and picklable run configuration """
        return self.provider.render_environment()

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
        print(f"Pytorch version: {torch.__version__} cuda version {torch.version.cuda} cudnn version {torch.backends.cudnn.version()}")  # noqa

        if self.tag:
            print("Running model {}, run {} ({}) -- command {} -- device {}".format(
                self._model_name, self.run_number, self.tag, command_name, self.device)
            )
        else:
            print("Running model {}, run {} -- command {} -- device {}".format(
                self._model_name, self.run_number, command_name, self.device)
            )

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

    ####################################################################################################################
    # CONVENIENCE METHODS FOR SCRIPTS
    def load_trained_model(self):
        """ Load a latest trained model from storage """
        model = self.provide("model").instantiate()
        storage = self.provide("storage")

        last_epoch_idx = storage.last_epoch_idx()

        if last_epoch_idx == 0:
            raise VelInitializationException("No trained model available")

        training_info = TrainingInfo(start_epoch_idx=last_epoch_idx)

        model_state, hidden_state = storage.load(training_info)

        model.load_state_dict(model_state)

        return model
