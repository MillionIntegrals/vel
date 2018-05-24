import pymongo

import waterboy.api.callback as cb


class MongoDBStorage(cb.Callback):
    """ Store model metrics to the database """

    def __init__(self, uri, database, model_config):
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[database]
        self.model_config = model_config

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

    def on_epoch_end(self, epoch_idx, metrics, model, optimizer):
        augmented_metrics = metrics.copy()

        model_name = self.model_config.name
        run_name = self.model_config.run_name

        augmented_metrics['model_name'] = model_name
        augmented_metrics['run_name'] = run_name

        self.db.metrics.insert_one(augmented_metrics)


def create(uri, database, model_config):
    """ Create a mongodb storage object """
    return MongoDBStorage(uri, database, model_config)
