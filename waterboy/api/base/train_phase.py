
class TrainPhase:
    """ A single phase of training """

    @property
    def number_of_epochs(self) -> int:
        """ How many epochs does this phase take """
        raise NotImplementedError

    def set_up_phase(self, learner, source, metrics=None, callbacks=None):
        """ Prepare the phase for learning """
        raise NotImplementedError

    def execute_epoch(self, epoch_idx, learner):
        """ Prepare the phase for learning """
        raise NotImplementedError

    def tear_down_phase(self, learner):
        """ Clean up after phase is done """
        raise NotImplementedError

    def restore(self, local_epoch_idx, learner, hidden_state):
        """
        Restore learning from intermediate state.
        """
        pass

    def state_dict(self):
        """
        State to save down
        """
        return {}
