import visdom
import pandas as pd

import waterboy.api.base as base

from waterboy.util.visdom import visdom_append_metrics


class VisdomStreaming(base.Callback):
    """ Stream live results to visdom from training """
    def __init__(self, model_config, stream_lr):
        self.model_config = model_config
        self.vis = visdom.Visdom(env=self.model_config.run_name)
        self.stream_lr = stream_lr

    def on_epoch_end(self, epoch_idx, metrics):
        """ Update data in visdom on push """
        metrics_df = pd.DataFrame([metrics]).set_index('epoch_idx')
        visdom_append_metrics(self.vis, metrics_df, first_epoch=epoch_idx.global_epoch_idx == 1)

    def on_batch_end(self, progress_idx,  _, optimizer):
        """ Stream LR to visdom """
        if self.stream_lr:
            iteration_idx = (
                    float(progress_idx.epoch_number) +
                    float(progress_idx.batch_number) / progress_idx.batches_per_epoch
            )
            
            lr = optimizer.param_groups[-1]['lr']

            metrics_df = pd.DataFrame([lr], index=[iteration_idx], columns=['lr'])

            visdom_append_metrics(
                self.vis, metrics_df,
                first_epoch=(progress_idx.epoch_number == 1) and (progress_idx.batch_number == 0)
            )


def create(model_config, stream_lr=False):
    return VisdomStreaming(model_config, stream_lr=stream_lr)
