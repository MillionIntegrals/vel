import pymongo
import pandas as pd


class MongoDbBackend:
    """ Storage backend to store all experiment results in a MongoDB database """

    def __init__(self, model_config, uri, database):
        self.model_config = model_config
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[database]

    def clean(self, initial_epoch):
        """ Remove entries from database that would get overwritten """
        self.db.metrics.delete_many({'run_name': self.model_config.run_name, 'epoch_idx': {'$gt': initial_epoch}})

    def store_config(self, configuration):
        """ Store model parameters in the database """
        run_name = self.model_config.run_name

        self.db.configs.delete_many({'run_name': self.model_config.run_name})

        configuration = configuration.copy()
        configuration['run_name'] = run_name

        self.db.configs.insert_one(configuration)

    def get_frame(self):
        """ Get a dataframe of metrics from this storage """
        metric_items = list(self.db.metrics.find({'run_name': self.model_config.run_name}).sort('epoch_idx'))
        if len(metric_items) == 0:
            return pd.DataFrame(columns=['run_name'])
        else:
            return pd.DataFrame(metric_items).drop(['_id', 'model_name'], axis=1).set_index('epoch_idx')

    def store(self, metrics):
        augmented_metrics = metrics.copy()

        model_name = self.model_config.name
        run_name = self.model_config.run_name

        augmented_metrics['model_name'] = model_name
        augmented_metrics['run_name'] = run_name

        self.db.metrics.insert_one(augmented_metrics)


def create(model_config, uri, database):
    """ Vel factory function """
    return MongoDbBackend(model_config, uri, database)

