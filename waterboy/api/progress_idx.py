class EpochIdx:
    """
    Class representing epoch number.
    Global epoch index - number of epoch from start of training until now
    Local epoch index - number of epoch from start of current "phase" until now
    """

    def __init__(self, global_epoch_idx: int, local_epoch_idx: int = None):
        self.global_epoch_idx = global_epoch_idx
        
        if local_epoch_idx is None:
            self.local_epoch_idx = self.global_epoch_idx
        else:
            self.local_epoch_idx = local_epoch_idx

    def __repr__(self):
        return f"EpochIdx(global_epoch_idx={self.global_epoch_idx}, local_epoch_idx={self.local_epoch_idx})"


class BatchIdx:
    """ Class representing how far are we in the training of the model """
    def __init__(self, epoch_idx: EpochIdx, batch_number: int, batches_per_epoch: int):
        self.epoch_number = epoch_idx.global_epoch_idx
        self.local_epoch_number = epoch_idx.local_epoch_idx
        self.batch_number = batch_number
        self.batches_per_epoch = batches_per_epoch
