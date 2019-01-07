import visdom
import pandas as pd


from vel.api import ModelConfig, Callback
from vel.util.visdom import visdom_append_metrics, VisdomSettings


class VisdomStreaming(Callback):
    """ Stream live results to visdom from training """
    def __init__(self, model_config: ModelConfig, visdom_settings: VisdomSettings):
        self.model_config = model_config
        self.settings = visdom_settings

        self.vis = visdom.Visdom(
            server=visdom_settings.server,
            endpoint=visdom_settings.endpoint,
            port=visdom_settings.port,
            env=self.model_config.run_name.replace('/', '_')
        )

    def on_epoch_end(self, epoch_info):
        """ Update data in visdom on push """
        metrics_df = pd.DataFrame([epoch_info.result]).set_index('epoch_idx')

        visdom_append_metrics(
            self.vis,
            metrics_df,
            first_epoch=epoch_info.global_epoch_idx == 1
        )

    def on_batch_end(self, batch_info):
        """ Stream LR to visdom """
        if self.settings.stream_lr:
            iteration_idx = (
                    float(batch_info.epoch_number) +
                    float(batch_info.batch_number) / batch_info.batches_per_epoch
            )
            
            lr = batch_info.optimizer.param_groups[-1]['lr']

            metrics_df = pd.DataFrame([lr], index=[iteration_idx], columns=['lr'])

            visdom_append_metrics(
                self.vis,
                metrics_df,
                first_epoch=(batch_info.epoch_number == 1) and (batch_info.batch_number == 0)
            )


def create(model_config, visdom_settings):
    """ Vel factory function """
    return VisdomStreaming(model_config, VisdomSettings(**visdom_settings))
