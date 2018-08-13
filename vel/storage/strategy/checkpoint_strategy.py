import typing


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


