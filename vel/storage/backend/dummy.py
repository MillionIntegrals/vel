import pandas as pd


class DummyBackend:
    """ Storage backend to store all experiment results in a MongoDB database """

    def __init__(self):
        pass

    def clean(self, initial_epoch):
        """ Remove entries from database that would get overwritten """
        pass

    def get_frame(self):
        """ Get a dataframe of metrics from this storage """
        return pd.DataFrame()

    def store(self, metrics):
        """ Store metrics in a datastore """
        pass


def create():
    """ Create a mongodb storage object """
    return DummyBackend()
