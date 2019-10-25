import os
import shutil

from vel.api import ModelConfig, Callback, TrainingInfo, EpochInfo, Model
from torch.utils.tensorboard import SummaryWriter


class TensorboardStreaming(Callback):
    """ Stream results to tensorboard """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.logdir = self.model_config.output_dir('tensorboard', self.model_config.run_name)

    def on_train_begin(self, training_info: TrainingInfo, model: Model) -> None:
        """ Potentially cleanup previous runs """
        if training_info.start_epoch_idx == 0:
            if os.path.exists(self.logdir):
                shutil.rmtree(self.logdir)

    def on_epoch_end(self, epoch_info: EpochInfo):
        """ Push data to tensorboard on push """
        head_set = sorted({x.dataset for x in epoch_info.result.keys()})

        for head in head_set:
            if head is None:
                summary_writer = SummaryWriter(log_dir=os.path.join(self.logdir, "generic"))
            else:
                summary_writer = SummaryWriter(log_dir=os.path.join(self.logdir, head))

            for key, value in epoch_info.result.items():
                if key.dataset == head:
                    tag = '{}/{}'.format(key.scope, key.name)

                    summary_writer.add_scalar(
                        tag=tag,
                        scalar_value=value,
                        global_step=epoch_info.global_epoch_idx,
                    )

            summary_writer.close()


def create(model_config):
    """ Vel factory function """
    return TensorboardStreaming(model_config)
