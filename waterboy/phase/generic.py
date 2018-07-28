import waterboy.api.base as base


class GenericPhase(base.TrainPhase):
    """ Most generic phase of training """

    def __init__(self, lr, epochs, optimizer_factory, freeze=False):
        self.lr = lr
        self.epochs = epochs
        self.optimizer_factory = optimizer_factory
        self.freeze = freeze

        self._optimizer_instance = None
        self._source = None
        self._metrics = []
        self._callbacks = []

    @property
    def number_of_epochs(self) -> int:
        return self.epochs

    def set_up_phase(self, learner, source, metrics=None, callbacks=None):
        """ Prepare the phase for learning """
        if self.freeze:
            learner.model.freeze()

        self._optimizer_instance = self.optimizer_factory.instantiate(filter(lambda p: p.requires_grad, learner.model.parameters()))
        self._source = source

        if metrics is not None:
            self._metrics = metrics

        if callbacks is not None:
            self._callbacks = callbacks

    def execute_epoch(self, epoch_idx, learner):
        """ Prepare the phase for learning """
        for param_group in self._optimizer_instance.param_groups:
            param_group['lr'] = self.lr

        epoch_result = learner.run_epoch(
            epoch_idx, self._metrics, self._source, self._optimizer_instance, self._callbacks
        )

        return epoch_result

    def tear_down_phase(self, learner):
        """ Clean up after phase is done """
        pass


def create(lr, epochs, optimizer, freeze=False):
    """ Waterboy creation function """
    return GenericPhase(
        lr=lr,
        epochs=epochs,
        optimizer_factory=optimizer,
        freeze=freeze
    )
