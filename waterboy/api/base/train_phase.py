class TrainPhase:
    """ A single phase of training """

    @property
    def number_of_epochs(self) -> int:
        """ How many epochs does this phase take """
        raise NotImplementedError

    def set_up_phase(self, learner, source, metrics=None, callbacks=None):
        """ Prepare the phase for learning """
        pass

    def execute_epoch(self, epoch_idx, learner):
        """ Execute epoch training """
        raise NotImplementedError

    def tear_down_phase(self, learner):
        """ Clean up after phase is done """
        pass

    def restore(self, epoch_idx, learner, hidden_state):
        """
        Restore learning from intermediate state.
        """
        pass

    def state_dict(self):
        """
        State to save down
        """
        return {}

    def banner(self) -> str:
        """ Return banner for the phase """
        return f"|------> PHASE: {self.__class__.__name__} Length: {self.number_of_epochs}"


class EmptyTrainPhase(TrainPhase):
    """ A train phase that is a simple call, without any training """

    @property
    def number_of_epochs(self) -> int:
        """ How many epochs does this phase take """
        return 0

    def execute_epoch(self, epoch_idx, learner):
        """ Prepare the phase for learning """
        pass
