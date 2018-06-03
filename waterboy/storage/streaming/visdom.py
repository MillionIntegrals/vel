import visdom
import pandas as pd

import waterboy.api.base as base

from waterboy.util.visdom import visdom_append_metrics


class VisdomStreaming(base.Callback):
    """ Stream live results to visdom from training """
    def __init__(self, model_config):
        self.model_config = model_config
        self.vis = visdom.Visdom(env=self.model_config.run_name)

    def on_epoch_end(self, epoch_idx, metrics):
        """ Update data in visdom on push """
        metrics_df = pd.DataFrame([metrics]).set_index('epoch_idx')
        visdom_append_metrics(self.vis, metrics_df, first_epoch=epoch_idx == 1)


def create(model_config):
    return VisdomStreaming(model_config)
