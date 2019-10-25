import wandb
import yaml


from vel.api import ModelConfig, Callback, TrainingInfo, Model


class WandbStreaming(Callback):
    """ Stream live results from training to WandB """

    def __init__(self, model_config: ModelConfig, register_model: bool = False, write_hyperparams: bool = True):
        self.model_config = model_config
        self.project = self.model_config.provide('project_name')
        self.register_model = register_model
        self.write_hyperparams = write_hyperparams

    def on_train_begin(self, training_info: TrainingInfo, model: Model) -> None:
        wandb.init(
            job_type='train',
            project=self.project,
            dir=self.model_config.model_output_dir('wandb'),
            group=self.model_config.name,
            name=self.model_config.run_name,
            resume=training_info.start_epoch_idx > 0,
            tags=[self.model_config.tag] if self.model_config.tag else []
        )

        if self.register_model:
            wandb.watch(model)

        if self.write_hyperparams:
            path = self.model_config.model_output_dir('wandb', 'vel-config.yaml')
            with open(path, 'wt') as fp:
                yaml.dump(self.model_config.render_configuration(), fp)
                wandb.save(path)

    def on_epoch_end(self, epoch_info):
        """ Send data to wandb """
        result = {k.format(): v for k, v in epoch_info.result.items()}
        wandb.log(row=result, step=epoch_info.global_epoch_idx)


def create(model_config, register_model: bool = False):
    """ Vel factory function """
    return WandbStreaming(model_config, register_model=register_model)
