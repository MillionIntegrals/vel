from vel.api import TrainingInfo, EpochInfo, OptimizedModel
from vel.data.loader import DatasetLoader
from vel.train import TrainPhase


class GenericPhase(TrainPhase):
    """ Most generic phase of training """

    def __init__(self, lr, epochs, optimizer_factory):
        self.lr = lr
        self.epochs = epochs
        self.optimizer_factory = optimizer_factory

        self._optimizer_instance = None
        self._loader = None

    @property
    def number_of_epochs(self) -> int:
        return self.epochs

    def set_up_phase(self, training_info: TrainingInfo, model: OptimizedModel, loader: DatasetLoader):
        """ Prepare the phase for learning """
        self._optimizer_instance = model.create_optimizer(self.optimizer_factory)
        self._loader = loader

    def epoch_info(self, training_info: TrainingInfo, global_idx: int, local_idx: int) -> EpochInfo:
        """ Create Epoch info """
        return EpochInfo(
            training_info=training_info,
            global_epoch_idx=global_idx,
            local_epoch_idx=local_idx,
            batches_per_epoch=self._loader.size['train'],
            optimizer=self._optimizer_instance
        )

    def execute_epoch(self, epoch_info, learner):
        """ Prepare the phase for learning """
        for param_group in epoch_info.optimizer.param_groups:
            param_group['lr'] = self.lr

        epoch_result = learner.run_epoch(epoch_info, self._loader)

        return epoch_result


def create(lr, epochs, optimizer):
    """ Vel factory function """
    return GenericPhase(
        lr=lr,
        epochs=epochs,
        optimizer_factory=optimizer,
    )
