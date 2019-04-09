from vel.api import ModelConfig


def load_config(config_path, run_number=0, device='cuda:0'):
    """ Load a ModelConfig from filename """
    return ModelConfig.from_file(
        ModelConfig.from_project_directory(config_path),
        run_number=run_number,
        device=device
    )
