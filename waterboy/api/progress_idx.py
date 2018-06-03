

class ProgressIdx:
    """ Class representing how far are we in the training of the model """
    def __init__(self, epoch_number, batch_number, batches_per_epoch):
        self.epoch_number = epoch_number
        self.batch_number = batch_number
        self.batches_per_epoch = batches_per_epoch
