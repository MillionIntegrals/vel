import pandas as pd


class DummyBackend:
    """ Storage backend to store all experiment data in /dev/null """

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

    def store_config(self, configuration):
        """ Store model parameters """
        pass


def create():
    """ Vel factory function """
    return DummyBackend()
