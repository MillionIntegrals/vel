import pymongo
import pandas as pd

import waterboy.api.callback as cb


class MongoDBStorage(cb.Callback):
    """ Store model metrics to the database """

    def __init__(self, uri, database, model_config, update_visdom=False):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[database]
        self.model_config = model_config
        self.update_visdom = update_visdom

        if self.update_visdom:
            import visdom
            self.vis = visdom.Visdom(env=self.model_config.run_name)
        else:
            self.vis = None

    def clean(self, initial_epoch):
        """ Remove entries from database that would get overwritten """
        self.db.metrics.delete_many({'run_name': self.model_config.run_name, 'epoch_idx': {'$gt': initial_epoch}})

    def best_metric(self, last_epoch, metric_name, metric_mode):
        """ Find epoch that maximizes given metric """
        metrics = list(self.db.metrics.find({'run_name': self.model_config.run_name, 'epoch_idx': {'$lte': last_epoch}}))

        if metric_mode == 'min':
            element = min(metrics, key=lambda x: x[metric_name])
            return element[metric_name], element['epoch_idx']
        elif metric_mode == 'max':
            element = max(metrics, key=lambda x: x[metric_name])
            return element[metric_name], element['epoch_idx']
        else:
            raise NotImplementedError

    def get_metrics(self):
        """ Get a dataframe of metrics from this storage """
        metric_items = list(self.db.metrics.find({'run_name': self.model_config.run_name}).sort('epoch_idx'))
        return pd.DataFrame(metric_items).drop(['_id', 'model_name'], axis=1).set_index('epoch_idx')

    def on_epoch_end(self, epoch_idx, epoch_time, metrics, model, optimizer):
        augmented_metrics = metrics.copy()

        model_name = self.model_config.name
        run_name = self.model_config.run_name

        augmented_metrics['model_name'] = model_name
        augmented_metrics['run_name'] = run_name
        augmented_metrics['epoch_time'] = epoch_time

        self.db.metrics.insert_one(augmented_metrics)

        if self.update_visdom:
            self._visdom_update_fn(epoch_idx, augmented_metrics)

    def _visdom_update_fn(self, epoch_idx, metrics):
        """ Update data in visdom on push """
        from waterboy.commands.vis_store import visdom_append_metrics

        metrics_df = pd.DataFrame([metrics]).drop(['_id', 'model_name', 'run_name'], axis=1).set_index('epoch_idx')
        visdom_append_metrics(self.vis, metrics_df, first_epoch=epoch_idx == 1)


def create(uri, database, model_config, update_visdom=False):
    """ Create a mongodb storage object """
    return MongoDBStorage(uri, database, model_config, update_visdom=update_visdom)
