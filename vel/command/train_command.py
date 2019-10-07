import typing

import vel.api as api
import vel.data as data
import vel.train as train

from vel.metric.samples_per_sec import SamplesPerSec
from vel.callback.time_tracker import TimeTracker
from vel.callback.sample_tracker import SampleTracker


class SimpleTrainCommand:
    """ Very simple training command - just run the supplied generators """

    def __init__(self, epochs: int, model_config: api.ModelConfig, model_factory: api.ModuleFactory,
                 optimizer_factory: api.OptimizerFactory, scheduler_factory: typing.Optional[api.SchedulerFactory],
                 loader: data.DatasetLoader, storage: api.Storage,
                 callbacks: typing.Optional[typing.List[api.Callback]]):
        self.epochs = epochs
        self.model_config = model_config
        self.model_factory = model_factory

        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        self.loader = loader
        self.storage = storage
        self.callbacks = callbacks if callbacks is not None else []

    def run(self):
        """ Run the command with supplied configuration """
        device = self.model_config.torch_device()

        trainer = train.Trainer(device, self.model_factory.instantiate())
        optimizer = trainer.model.create_optimizer(self.optimizer_factory)

        # Check if training was already started and potentially continue where we left off
        training_info = self.start_training(trainer, optimizer)

        training_info.on_train_begin()

        for global_epoch_idx in range(training_info.start_epoch_idx + 1, self.epochs + 1):
            epoch_info = api.EpochInfo(
                training_info=training_info,
                global_epoch_idx=global_epoch_idx,
                batches_per_epoch=self.loader.size['train'],
                optimizer=optimizer
            )

            # Execute learning
            trainer.run_epoch(epoch_info, self.loader)

            self.storage.checkpoint(epoch_info, trainer.model)

        training_info.on_train_end()

        return training_info

    def start_training(self, trainer: train.Trainer, optimizer: api.VelOptimizer) -> api.TrainingInfo:
        """ Possibly resume training from a saved state from the storage """
        if self.model_config.resume_training:
            start_epoch = self.storage.last_epoch_idx()
        else:
            start_epoch = 0

        # Initial set of callbacks, always useful
        callbacks = [TimeTracker(), SampleTracker()]

        if self.scheduler_factory is not None:
            callbacks.extend(
                optimizer.create_scheduler(scheduler_factory=self.scheduler_factory, last_epoch=start_epoch-1)
            )

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        # Metrics to track through this training
        metrics = trainer.metrics() + optimizer.metrics() + [SamplesPerSec()]

        training_info = api.TrainingInfo(
            start_epoch_idx=start_epoch,
            metrics=metrics,
            callbacks=callbacks
        )

        if start_epoch == 0:
            self.model_config.write_meta()
            self.storage.reset(self.model_config.render_configuration())
            training_info.initialize()
            trainer.initialize_training(training_info)
        else:
            model_state, hidden_state = self.storage.load(training_info)

            training_info.restore(hidden_state)
            trainer.initialize_training(training_info, model_state, hidden_state)

            if 'optimizer' in hidden_state:
                optimizer.load_state_dict(hidden_state['optimizer'])

        return training_info


def create(model_config, epochs, optimizer, model, loader, storage, scheduler=None, callbacks=None):
    """ Vel factory function """
    return SimpleTrainCommand(
        epochs=epochs,
        model_config=model_config,
        model_factory=model,
        optimizer_factory=optimizer,
        scheduler_factory=scheduler,
        loader=loader,
        storage=storage,
        callbacks=callbacks,
    )
