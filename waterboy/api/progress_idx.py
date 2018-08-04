class EpochIdx:
    """
    Class representing epoch number.
    Global epoch index - number of epoch from start of training until now
    Local epoch index - number of epoch from start of current "phase" until now

    extra - any additional information anyone would like to put next to epoch information
    """

    def __init__(self, global_epoch_idx: int, local_epoch_idx: int = None, extra: dict=None):
        self.global_epoch_idx = global_epoch_idx
        
        if local_epoch_idx is None:
            self.local_epoch_idx = self.global_epoch_idx
        else:
            self.local_epoch_idx = local_epoch_idx

        self.extra = extra

    def __repr__(self):
        return f"EpochIdx(global_epoch_idx={self.global_epoch_idx}, local_epoch_idx={self.local_epoch_idx})"


class BatchIdx:
    """ Class representing how far are we in the training of the model """
    def __init__(self, epoch_idx: EpochIdx, batch_number: int, batches_per_epoch: int, extra: dict=None):
        self.epoch_idx = epoch_idx

        self.batch_number = batch_number
        self.batches_per_epoch = batches_per_epoch

        self.extra = extra

    @property
    def epoch_number(self):
        return self.epoch_idx.global_epoch_idx

    @property
    def local_epoch_number(self):
        return self.epoch_idx.local_epoch_idx

    @property
    def epoch_extra(self):
        return self.epoch_idx.extra
