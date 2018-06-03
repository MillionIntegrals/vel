import typing


from waterboy.util.better import better


class CheckpointStrategy:
    """ Base class for various checkpoint strategies """
    def should_delete_previous_checkpoint(self, epoch_idx) -> bool:
        """ Should previous checkpoint be deleted or stored """
        return True

    def should_store_best_checkpoint(self, epoch_idx, metrics) -> bool:
        """ Should we store current checkpoint as the best """
        return False

    def store_best_checkpoint_idx(self, epoch_idx) -> None:
        """ Should we store current checkpoint as the best """
        pass

    @property
    def current_best_checkpoint_idx(self) -> typing.Union[int, None]:
        return None

    def write_state_dict(self, hidden_state_dict): pass

    def restore(self, hidden_state_dict): pass


class ClassicCheckpointStrategy(CheckpointStrategy):
    """ Classic checkpoint strategy """
    def __init__(self, checkpoint_frequency=0, metric=None, metric_mode='min', store_best=False):
        self.checkpoint_frequency = checkpoint_frequency
        self.metric = metric or 'val:loss'
        self.metric_mode = metric_mode
        self.store_best = store_best

        # TODO(jerry) initialize these values from hidden state
        self._current_best_metric_value = None
        self._current_best_checkpoint_idx = None

    def should_delete_previous_checkpoint(self, epoch_idx) -> bool:
        prev_epoch_idx = epoch_idx - 1

        if self.checkpoint_frequency > 0 and prev_epoch_idx % self.checkpoint_frequency == 0:
            return False
        else:
            return True

    def should_store_best_checkpoint(self, epoch_idx, metrics) -> bool:
        """ Should we store current checkpoint as the best """
        if not self.store_best:
            return False

        metric = metrics[self.metric]

        if better(self._current_best_metric_value, metric, self.metric_mode):
            self._current_best_metric_value = metric
            return True

        return False

    def store_best_checkpoint_idx(self, epoch_idx) -> None:
        """ Should we store current checkpoint as the best """
        self._current_best_checkpoint_idx = epoch_idx

    @property
    def current_best_checkpoint_idx(self) -> typing.Union[int, None]:
        return self._current_best_checkpoint_idx
