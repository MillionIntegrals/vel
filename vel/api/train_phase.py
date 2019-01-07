from torch.optim import Optimizer

from vel.api import TrainingInfo, EpochInfo, Learner, Model, Source


class TrainPhase:
    """ A single phase of training """

    @property
    def number_of_epochs(self) -> int:
        """ How many epochs does this phase take """
        raise NotImplementedError

    def set_up_phase(self, training_info: TrainingInfo, model: Model, source: Source) -> Optimizer:
        """ Prepare the phase for learning, returns phase optimizer """
        pass

    def restore(self, training_info: TrainingInfo, local_batch_idx: int, model: Model, hidden_state: dict):
        """
        Restore learning from intermediate state.
        """
        pass

    def epoch_info(self, training_info: TrainingInfo, global_idx: int, local_idx: int) -> EpochInfo:
        """ Create Epoch info """
        raise NotImplementedError

    def execute_epoch(self, epoch_info: EpochInfo, learner: Learner):
        """
        Execute epoch training.
        """
        raise NotImplementedError

    def tear_down_phase(self, training_info: TrainingInfo, model: Model):
        """ Clean up after phase is done """
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

    def execute_epoch(self, epoch_info, learner):
        """ Prepare the phase for learning """
        pass

    def epoch_info(self, training_info: TrainingInfo, global_idx: int, local_idx: int) -> EpochInfo:
        """ Create Epoch info """
        return EpochInfo(training_info, global_epoch_idx=global_idx, local_epoch_idx=local_idx, batches_per_epoch=0)
