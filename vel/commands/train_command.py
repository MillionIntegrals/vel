import torch
import typing

from vel.api import Learner, ModelConfig, EpochInfo, TrainingInfo
from vel.api.base import OptimizerFactory, SchedulerFactory, Callback, Source, Storage, ModelFactory
from vel.callbacks.time_tracker import TimeTracker


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, epochs: int, model_config: ModelConfig, model_factory: ModelFactory,
                 optimizer_factory: OptimizerFactory, scheduler_factory: typing.Optional[SchedulerFactory],
                 source: Source, storage: Storage, callbacks: typing.Optional[typing.List[Callback]],
                 max_grad_norm: typing.Optional[float]):
        self.epochs = epochs
        self.model_config = model_config
        self.model_factory = model_factory

        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        self.source = source
        self.storage = storage
        self.callbacks = callbacks if callbacks is not None else []
        self.max_grad_norm = max_grad_norm

    def run(self):
        """ Run the command with supplied configuration """
        device = torch.device(self.model_config.device)
        learner = Learner(device, self.model_factory.instantiate(), self.max_grad_norm)
        optimizer = self.optimizer_factory.instantiate(learner.model)

        # All callbacks used for learning
        callbacks = self.gather_callbacks(optimizer)

        # Metrics to track through this training
        metrics = learner.metrics()

        # Check if training was already started and potentially continue where we left off
        training_info = self.resume_training(learner, callbacks, metrics)

        training_info.on_train_begin()

        if training_info.optimizer_initial_state:
            optimizer.load_state_dict(training_info.optimizer_initial_state)

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

        training_info.on_train_end()

        return training_info

    def gather_callbacks(self, optimizer) -> list:
        """ Gather all the callbacks to be used in this training run """
        callbacks = [TimeTracker()]

        if self.scheduler_factory is not None:
            callbacks.append(self.scheduler_factory.instantiate(optimizer))

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        return callbacks

    def resume_training(self, learner, callbacks, metrics) -> TrainingInfo:
        """ Possibly resume training from a saved state from the storage """
        if self.model_config.continue_training:
            start_epoch = self.storage.last_epoch_idx()
        else:
            start_epoch = 0

        training_info = TrainingInfo(
            start_epoch_idx=start_epoch,
            run_name=self.model_config.run_name,
            metrics=metrics,
            callbacks=callbacks
        )

        if start_epoch == 0:
            self.storage.reset(self.model_config.render_configuration())
            training_info.initialize()
            learner.initialize_training(training_info)
        else:
            self.storage.resume(training_info, learner.model)

        return training_info


def create(model_config, epochs, optimizer, model, source, storage, scheduler=None, callbacks=None, max_grad_norm=None):
    """ Simply train the model """
    return SimpleTrainCommand(
        epochs=epochs,
        model_config=model_config,
        model_factory=model,
        optimizer_factory=optimizer,
        scheduler_factory=scheduler,
        source=source,
        storage=storage,
        callbacks=callbacks,
        max_grad_norm=max_grad_norm
    )
