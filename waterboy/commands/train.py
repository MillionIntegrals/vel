import torch

from waterboy.api import Learner, ModelConfig, EpochIdx
from waterboy.api.metrics import TrainingHistory
from waterboy.storage.impl.checkpoint_strategy import ClassicCheckpointStrategy


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, model_config: ModelConfig, epochs, optimizer_fn, scheduler_fn, callbacks, checkpoint, model,
                 source, storage, restart=True):
        self.epochs = epochs
        self.callbacks = callbacks
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.checkpoint = checkpoint
        self.model = model
        self.source = source
        self.model_config = model_config
        self.storage = storage
        self.restart = restart

        self.storage.set_checkpoint_strategy(ClassicCheckpointStrategy(**self.checkpoint))

    def restore(self, hidden_state, optimizer, callbacks):
        optimizer.load_state_dict(hidden_state['optimizer'])

        for callback in callbacks:
            callback.load_state_dict(hidden_state)

    def run(self):
        """ Run the command with supplied configuration """
        device = torch.device(self.model_config.device)
        learner = Learner(device, self.model)

        optimizer_instance = self.optimizer_fn(learner.model.parameters())

        callbacks = []

        if self.scheduler_fn is not None:
            callbacks.append(self.scheduler_fn(optimizer_instance))

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        # Just default set of model metrics
        metrics = learner.metrics()

        if self.restart:
            last_epoch, hidden_state = self.storage.resume_learning(learner.model)
        else:
            last_epoch, hidden_state = 0, {}

        if last_epoch > 0:
            self.restore(hidden_state, optimizer_instance, callbacks)

        for callback in callbacks:
            callback.on_train_begin()

        training_history = TrainingHistory()

        for global_epoch_idx in range(1 + last_epoch, self.epochs+1):
            epoch_idx = EpochIdx(global_epoch_idx)
            epoch_result = learner.run_epoch(epoch_idx, metrics, self.source, optimizer_instance, callbacks)

            self.storage.checkpoint(epoch_idx, epoch_result, learner.model, optimizer_instance, callbacks)

            training_history.add(epoch_result)

        for callback in callbacks:
            callback.on_train_end()

        return training_history


def create(model_config, epochs, optimizer, model, source,  storage, scheduler=None, callbacks=None, checkpoint=None,
           restart=True):
    """ Simply train the model """
    callbacks = callbacks or []
    checkpoint = checkpoint or {}

    return SimpleTrainCommand(
        model_config=model_config,
        epochs=epochs,
        optimizer_fn=optimizer,
        scheduler_fn=scheduler,
        callbacks=callbacks,
        checkpoint=checkpoint,
        model=model,
        source=source,
        storage=storage,
        restart=restart
    )
