import wandb


from vel.api import ModelConfig, Callback, TrainingInfo


class WandbStreaming(Callback):
    """ Stream live results from training to WandB """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def on_train_begin(self, training_info: TrainingInfo) -> None:
        wandb.init(
            job_type='train',
            project='vel',
            dir=self.model_config.output_dir('wandb'),
            group=self.model_config.name,
            name=self.model_config.run_name,
            resume=training_info.start_epoch_idx > 0,
            tags=[self.model_config.tag] if self.model_config.tag else []
        )

    def on_epoch_end(self, epoch_info):
        """ Send data to wandb """
        result = {k.format(): v for k, v in epoch_info.result.items()}
        wandb.log(row=result, step=epoch_info.global_epoch_idx)


def create(model_config):
    """ Vel factory function """
    return WandbStreaming(model_config)
