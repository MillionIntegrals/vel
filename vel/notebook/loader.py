from vel.api import ModelConfig


def load(config_path, run_number=0, device='cuda:0'):
    """ Load a ModelConfig from filename """
    model_config = ModelConfig.from_file(config_path, run_number, device=device)

    return model_config
