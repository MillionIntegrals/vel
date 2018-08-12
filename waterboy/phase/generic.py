import waterboy.api.base as base


class GenericPhase(base.TrainPhase):
    """ Most generic phase of training """

    def __init__(self, lr, epochs, optimizer_factory):
        self.lr = lr
        self.epochs = epochs
        self.optimizer_factory = optimizer_factory

        self._optimizer_instance = None
        self._source = None

    @property
    def number_of_epochs(self) -> int:
        return self.epochs

    def set_up_phase(self, training_info, model, source):
        """ Prepare the phase for learning """
        self._optimizer_instance = self.optimizer_factory.instantiate(filter(lambda p: p.requires_grad, model.parameters()))
        self._source = source
        return self._optimizer_instance

    def execute_epoch(self, epoch_info, learner):
        """ Prepare the phase for learning """
        for param_group in epoch_info.optimizer.param_groups:
            param_group['lr'] = self.lr

        epoch_result = learner.run_epoch(epoch_info, self._source)

        return epoch_result


def create(lr, epochs, optimizer):
    """ Waterboy creation function """
    return GenericPhase(
        lr=lr,
        epochs=epochs,
        optimizer_factory=optimizer,
    )
