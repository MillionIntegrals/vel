import torch

from waterboy.api import ModelConfig, EpochIdx
from waterboy.api.base import Model, OptimizerFactory, Storage
from waterboy.api.metrics import EpochResultAccumulator, TrainingHistory
from waterboy.rl.api.base import ReinforcerFactory

import waterboy.openai.baselines.logger as openai_logger


class RlTrainCommand:
    """ Train a reinforcement learning algorithm by evaluating the environment and """
    def __init__(self, model_config: ModelConfig, reinforcer: ReinforcerFactory,
                 model: Model, optimizer_factory: OptimizerFactory,
                 storage: Storage, callbacks,
                 total_frames: int, batches_per_epoch: int, seed: int,
                 scheduler_factory=None):
        self.model_config = model_config
        self.reinforcer = reinforcer
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.storage = storage
        self.total_frames = total_frames
        self.batches_per_epoch = batches_per_epoch
        self.callbacks = callbacks

        self.seed = seed

    def run(self):
        """ Run reinforcement learning algorithm """
        device = torch.device(self.model_config.device)

        total_framecount = 0
        global_epoch_idx = 1

        # Reinforcer is the learner for the reinforcement learning model
        reinforcer = self.reinforcer.instantiate(device, self.model)
        reinforcer.model.reset_weights()

        optimizer_instance = self.optimizer_factory.instantiate(reinforcer.model.parameters())
        metrics = reinforcer.metrics()

        callbacks = []

        if self.scheduler_factory is not None:
            callbacks.append(self.scheduler_factory.instantiate(optimizer_instance))

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        for callback in callbacks:
            callback.on_train_begin()

        training_history = TrainingHistory()

        while total_framecount < self.total_frames:
            epoch_idx = EpochIdx(global_epoch_idx, extra={'total_frames': self.total_frames})
            result_accumulator = EpochResultAccumulator(epoch_idx, metrics)

            epoch_result = reinforcer.train_epoch(
                epoch_idx,
                batches_per_epoch=self.batches_per_epoch,
                optimizer=optimizer_instance,
                callbacks=callbacks,
                result_accumulator=result_accumulator
            )

            # OpenAI logging..., possibly guard it with a flag?
            for key in sorted(epoch_result.keys()):
                if key == 'fps':
                    # Not super elegant, but I like nicer display of FPS
                    openai_logger.record_tabular(key, int(epoch_result[key]))
                else:
                    openai_logger.record_tabular(key, epoch_result[key])

            openai_logger.dump_tabular()

            self.storage.checkpoint(
                epoch_idx.global_epoch_idx, epoch_result, reinforcer.model, optimizer_instance, callbacks
            )

            training_history.add(epoch_result)

            total_framecount = epoch_result['frames']
            global_epoch_idx += 1

        for callback in callbacks:
            callback.on_train_end()

        return training_history


def create(model_config, reinforcer, model, optimizer, storage,
           # Settings:
           total_frames, batches_per_epoch,  seed, callbacks=None, scheduler=None):
    """ Create reinforcement learning pipeline """
    callbacks = callbacks or []

    return RlTrainCommand(
        model_config=model_config,
        reinforcer=reinforcer,
        model=model,
        optimizer_factory=optimizer,
        scheduler_factory=scheduler,
        storage=storage,
        callbacks=callbacks,
        total_frames=int(float(total_frames)),
        batches_per_epoch=int(batches_per_epoch),
        seed=seed
    )
