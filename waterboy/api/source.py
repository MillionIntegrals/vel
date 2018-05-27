class Source:
    """ Very simple wrapper for a training and validation datasource """
    def __init__(self, train_source, val_source):
        self.train_source = train_source
        self.val_source = val_source

    def train_dataset(self):
        """ Return the training dataset """
        return self.train_source.dataset

    def val_dataset(self):
        """ Return the validation dataset """
        return self.val_source.dataset

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        return len(self.train_source)

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        return len(self.val_source)
