import torch

from waterboy.api import Learner, ModelConfig, EpochIdx
from waterboy.api.metrics import TrainingHistory


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, model_config: ModelConfig, model_factory, epochs, optimizer_factory, scheduler_factory, callbacks,
                 source, storage):
        self.epochs = epochs
        self.callbacks = callbacks
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.model = model_factory
        self.source = source
        self.model_config = model_config
        self.storage = storage

    def restore(self, hidden_state, optimizer, callbacks):
        optimizer.load_state_dict(hidden_state['optimizer'])

        for callback in callbacks:
            callback.load_state_dict(hidden_state)

    def run(self):
        """ Run the command with supplied configuration """
        device = torch.device(self.model_config.device)
        learner = Learner(device, self.model.instantiate())

        optimizer_instance = self.optimizer_factory.instantiate(learner.model.parameters())

        callbacks = []

        if self.scheduler_factory is not None:
            callbacks.append(self.scheduler_factory.instantiate(optimizer_instance))

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        # Just default set of model metrics
        metrics = learner.metrics()

        if self.model_config.reset:
            last_epoch, hidden_state = 0, {}
        else:
            last_epoch, hidden_state = self.storage.resume_learning(learner.model)

        if last_epoch > 0:
            self.restore(hidden_state, optimizer_instance, callbacks)

        for callback in callbacks:
            callback.on_train_begin()

        training_history = TrainingHistory()

        for global_epoch_idx in range(1 + last_epoch, self.epochs+1):
            epoch_idx = EpochIdx(global_epoch_idx)
            epoch_result = learner.run_epoch(epoch_idx, metrics, self.source, optimizer_instance, callbacks)

            self.storage.checkpoint(epoch_idx.global_epoch_idx, epoch_result, learner.model, optimizer_instance, callbacks)

            training_history.add(epoch_result)

        for callback in callbacks:
            callback.on_train_end()

        return training_history


def create(model_config, epochs, optimizer, model, source, storage, scheduler=None, callbacks=None):
    """ Simply train the model """
    callbacks = callbacks or []

    return SimpleTrainCommand(
        model_config=model_config,
        model_factory=model,
        epochs=epochs,
        optimizer_factory=optimizer,
        scheduler_factory=scheduler,
        callbacks=callbacks,
        source=source,
        storage=storage,
    )
