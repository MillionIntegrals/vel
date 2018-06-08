from waterboy.internals.project_config import ProjectConfig
from waterboy.api import ModelConfig


def load(config_path, run_number=0, device='cuda:0'):
    """ Load a ModelConfig from filename """

    project_config = ProjectConfig(config_path)
    model_config = ModelConfig(config_path, run_number, project_config, device=device)

    return model_config
