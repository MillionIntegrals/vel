import vel.api as api


class Normalize(api.ScopedTransformation):
    """ Normalize input mean and standard deviation """

    def __init__(self, scope='x', tags=None):
        super().__init__(scope, tags)

        self.mean = None
        self.std = None

    def initialize(self, source):
        """ Initialize transformation from source """
        self.mean = source.metadata['train_mean']
        self.std = source.metadata['train_std']

    def transform(self, value):
        return (value - self.mean) / self.std

    def denormalization_transform(self, value):
        """ Operation reverse to normalization """
        return value * self.std + self.mean


def create(mode='x', tags=None):
    """ Vel factory function """
    return Normalize(scope=mode, tags=tags)
