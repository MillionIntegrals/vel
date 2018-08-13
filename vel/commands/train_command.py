import torch

from vel.api import Learner, ModelConfig, EpochInfo, TrainingInfo


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, model_config: ModelConfig, model_factory, epochs, optimizer_factory, scheduler_factory,
                 callbacks, source, storage):
        self.epochs = epochs
        self.callbacks = callbacks
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.model_factory = model_factory
        self.source = source
        self.model_config = model_config
        self.storage = storage

    def run(self):
        """ Run the command with supplied configuration """
        device = torch.device(self.model_config.device)
        learner = Learner(device, self.model_factory.instantiate())
        optimizer = self.optimizer_factory.instantiate(learner.model.parameters())

        # All callbacks used for learning
        callbacks = self.gather_callbacks(optimizer)

        # Metrics to track through this training
        metrics = learner.metrics()

        # Check if training was already started and potentially continue where we left off
        training_info = self.resume_training(learner, optimizer, callbacks, metrics)

        for callback in callbacks:
            callback.on_train_begin(training_info)

        for global_epoch_idx in range(training_info.start_epoch_idx + 1, self.epochs + 1):
            epoch_info = EpochInfo(
                training_info=training_info,
                global_epoch_idx=global_epoch_idx,
                batches_per_epoch=self.source.train_iterations_per_epoch(),
                optimizer=optimizer
            )

            # Execute learning
            learner.run_epoch(epoch_info, self.source)

            self.storage.checkpoint(epoch_info, learner.model)

            training_info.history.add(epoch_info.result)

        for callback in callbacks:
            callback.on_train_end(training_info)

        return training_info

    def gather_callbacks(self, optimizer) -> list:
        """ Gather all the callbacks to be used in this training run """
        callbacks = []

        if self.scheduler_factory is not None:
            callbacks.append(self.scheduler_factory.instantiate(optimizer))

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        return callbacks

    def resume_training(self, learner, optimizer, callbacks, metrics) -> TrainingInfo:
        """ Possibly resume training from a saved state from the storage """
        if self.model_config.reset:
            start_epoch, hidden_state = 0, {}
        else:
            start_epoch, hidden_state = self.storage.resume_learning(learner.model)

        training_info = TrainingInfo(start_epoch_idx=start_epoch, metrics=metrics, callbacks=callbacks)

        if start_epoch > 0:
            self.restore_state(hidden_state, optimizer, callbacks)
            training_info.restore(hidden_state)

        return training_info

    def restore_state(self, hidden_state, optimizer, callbacks):
        """ Load state into optimizer and callbacks """
        optimizer.load_state_dict(hidden_state['optimizer'])

        for callback in callbacks:
            callback.load_state_dict(hidden_state)


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
