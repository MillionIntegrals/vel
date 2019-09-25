from vel.api import ModelConfig


def load_config(config_path, run_number=0, device='cuda:0', continue_training=True):
    """ Load a ModelConfig from filename """
    return ModelConfig.from_file(
        ModelConfig.from_project_directory(config_path),
        run_number=run_number,
        device=device,
        continue_training=continue_training
    )


def script(model_name: str = 'script', run_number=0, device='cuda:0', continue_training=True):
    """ Create an ad-hoc script model config """
    return ModelConfig.script(
        model_name=model_name,
        run_number=run_number,
        device=device,
        continue_training=continue_training
    )
