import os
import shutil

from vel.api import ModelConfig, Callback, TrainingInfo
from torch.utils.tensorboard import SummaryWriter


class TensorboardStreaming(Callback):
    """ Stream results to tensorboard """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.logdir = self.model_config.output_dir('tensorboard', self.model_config.run_name)

    def on_train_begin(self, training_info: TrainingInfo) -> None:
        """ Potentially cleanup previous runs """
        if training_info.start_epoch_idx == 0:
            if os.path.exists(self.logdir):
                shutil.rmtree(self.logdir)

    def on_epoch_end(self, epoch_info):
        """ Push data to tensorboard on push """
        summary_writer = SummaryWriter(log_dir=self.logdir)

        for key, value in epoch_info.result.items():
            if key == 'epoch_idx':
                continue

            summary_writer.add_scalar(
                tag=key,
                scalar_value=value,
                global_step=epoch_info.global_epoch_idx,
            )

        summary_writer.close()


def create(model_config):
    """ Vel factory function """
    return TensorboardStreaming(model_config)
