import torch

from vel.api import ModelConfig, EpochInfo, TrainingInfo, BatchInfo
from vel.api.base import OptimizerFactory, Storage, Callback
from vel.rl.api.base import ReinforcerFactory
from vel.callbacks.time_tracker import TimeTracker

import vel.openai.baselines.logger as openai_logger


class FrameTracker(Callback):
    """ Aggregate frame count from each batch to a global number """
    def on_train_begin(self, training_info: TrainingInfo):
        training_info['frames'] = 0

    def on_batch_begin(self, batch_info: BatchInfo):
        if 'total_frames' in batch_info.training_info:
            # Track progress during learning
            batch_info['progress'] = (
                    batch_info.training_info['frames'] / batch_info.training_info['total_frames']
            )

    def on_batch_end(self, batch_info: BatchInfo):
        batch_info.training_info['frames'] += batch_info['frames'].item()


class RlTrainCommand:
    """ Train a reinforcement learning algorithm by evaluating the environment and """
    def __init__(self, model_config: ModelConfig, reinforcer: ReinforcerFactory,
                 optimizer_factory: OptimizerFactory,
                 storage: Storage, callbacks,
                 total_frames: int, batches_per_epoch: int, seed: int,
                 scheduler_factory=None, openai_logging=False):
        self.model_config = model_config
        self.reinforcer = reinforcer
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.storage = storage
        self.total_frames = total_frames
        self.batches_per_epoch = batches_per_epoch
        self.callbacks = callbacks

        self.seed = seed
        self.openai_logging = openai_logging

    def run(self):
        """ Run reinforcement learning algorithm """
        device = torch.device(self.model_config.device)
        # Reinforcer is the learner for the reinforcement learning model
        reinforcer = self.reinforcer.instantiate(device)
        optimizer = self.optimizer_factory.instantiate(reinforcer.model.parameters())

        # All callbacks used for learning
        callbacks = self.gather_callbacks(optimizer)

        # Metrics to track through this training
        metrics = reinforcer.metrics()

        training_info = self.resume_training(reinforcer, optimizer, callbacks, metrics)

        reinforcer.initialize_training()

        for callback in callbacks:
            callback.on_train_begin(training_info)

        global_epoch_idx = training_info.start_epoch_idx
        training_info['total_frames'] = self.total_frames

        while training_info['frames'] < self.total_frames:
            epoch_info = EpochInfo(
                training_info,
                global_epoch_idx=global_epoch_idx,
                batches_per_epoch=self.batches_per_epoch,
                optimizer=optimizer,
            )

            reinforcer.train_epoch(epoch_info)

            if self.openai_logging:
                self._openai_logging(epoch_info.result)

            self.storage.checkpoint(epoch_info, reinforcer.model)

            training_info.history.add(epoch_info.result)

            global_epoch_idx += 1

        for callback in callbacks:
            callback.on_train_end(training_info)

        return training_info

    def gather_callbacks(self, optimizer) -> list:
        """ Gather all the callbacks to be used in this training run """
        callbacks = [FrameTracker(), TimeTracker()]

        if self.scheduler_factory is not None:
            callbacks.append(self.scheduler_factory.instantiate(optimizer))

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        return callbacks

    def resume_training(self, reinforcer, optimizer, callbacks, metrics) -> TrainingInfo:
        """ Possibly resume training from a saved state from the storage """
        global_epoch_idx = 1

        # TODO(jerry): Implement training resume
        training_info = TrainingInfo(start_epoch_idx=global_epoch_idx, metrics=metrics, callbacks=callbacks)

        return training_info

    def _openai_logging(self, epoch_result):
        for key in sorted(epoch_result.keys()):
            if key == 'fps':
                # Not super elegant, but I like nicer display of FPS
                openai_logger.record_tabular(key, int(epoch_result[key]))
            else:
                openai_logger.record_tabular(key, epoch_result[key])

        openai_logger.dump_tabular()


def create(model_config, reinforcer, optimizer, storage,
           # Settings:
           total_frames, batches_per_epoch,  seed, callbacks=None, scheduler=None, openai_logging=False):
    """ Create reinforcement learning pipeline """
    callbacks = callbacks or []

    from vel.openai.baselines import logger
    logger.configure(dir=model_config.openai_dir())

    return RlTrainCommand(
        model_config=model_config,
        reinforcer=reinforcer,
        optimizer_factory=optimizer,
        scheduler_factory=scheduler,
        storage=storage,
        callbacks=callbacks,
        total_frames=int(float(total_frames)),
        batches_per_epoch=int(batches_per_epoch),
        seed=seed,
        openai_logging=openai_logging
    )
