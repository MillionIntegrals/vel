import typing

from vel.api import ModelConfig, EpochInfo, TrainingInfo, BatchInfo, OptimizerFactory, Storage, Callback
from vel.rl.api import ReinforcerFactory
from vel.callbacks.time_tracker import TimeTracker

import vel.openai.baselines.logger as openai_logger


class FrameTracker(Callback):
    """ Aggregate frame count from each batch to a global number """
    def __init__(self, max_frames: typing.Optional[typing.Union[int, float]] = None):
        self.max_frames = max_frames

    def on_initialization(self, training_info: TrainingInfo):
        if self.max_frames is not None:
            training_info['total_frames'] = int(self.max_frames)

        training_info['frames'] = 0

    def on_batch_begin(self, batch_info: BatchInfo):
        if 'total_frames' in batch_info.training_info:
            # Track progress during learning
            batch_info['progress'] = (
                float(batch_info.training_info['frames']) / batch_info.training_info['total_frames']
            )

    def on_batch_end(self, batch_info: BatchInfo):
        batch_info.training_info['frames'] += batch_info['frames']

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        hidden_state_dict['frame_tracker/frames'] = training_info['frames']

        if 'total_frames' in training_info:
            hidden_state_dict['frame_tracker/total_frames'] = training_info['total_frames']

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        training_info['frames'] = hidden_state_dict['frame_tracker/frames']

        if 'frame_tracker/total_frames' in hidden_state_dict:
            training_info['total_frames'] = hidden_state_dict['frame_tracker/total_frames']


class RlTrainCommand:
    """ Train a reinforcement learning algorithm by evaluating the environment and """
    def __init__(self, model_config: ModelConfig, reinforcer: ReinforcerFactory,
                 optimizer_factory: OptimizerFactory,
                 storage: Storage, callbacks,
                 total_frames: int, batches_per_epoch: int,
                 scheduler_factory=None, openai_logging=False):
        self.model_config = model_config
        self.reinforcer = reinforcer
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.storage = storage
        self.total_frames = total_frames
        self.batches_per_epoch = batches_per_epoch
        self.callbacks = callbacks if callbacks is not None else []

        self.openai_logging = openai_logging

    def run(self):
        """ Run reinforcement learning algorithm """
        device = self.model_config.torch_device()

        # Reinforcer is the learner for the reinforcement learning model
        reinforcer = self.reinforcer.instantiate(device)
        optimizer = self.optimizer_factory.instantiate(reinforcer.model)

        # All callbacks used for learning
        callbacks = self.gather_callbacks(optimizer)
        # Metrics to track through this training
        metrics = reinforcer.metrics()

        training_info = self.resume_training(reinforcer, callbacks, metrics)

        reinforcer.initialize_training(training_info)
        training_info.on_train_begin()

        if training_info.optimizer_initial_state:
            optimizer.load_state_dict(training_info.optimizer_initial_state)

        global_epoch_idx = training_info.start_epoch_idx + 1

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

            global_epoch_idx += 1

        training_info.on_train_end()

        return training_info

    def gather_callbacks(self, optimizer) -> list:
        """ Gather all the callbacks to be used in this training run """
        callbacks = [FrameTracker(self.total_frames), TimeTracker()]

        if self.scheduler_factory is not None:
            callbacks.append(self.scheduler_factory.instantiate(optimizer))

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        return callbacks

    def resume_training(self, reinforcer, callbacks, metrics) -> TrainingInfo:
        """ Possibly resume training from a saved state from the storage """
        if self.model_config.continue_training:
            start_epoch = self.storage.last_epoch_idx()
        else:
            start_epoch = 0

        training_info = TrainingInfo(
            start_epoch_idx=start_epoch,
            run_name=self.model_config.run_name,
            metrics=metrics, callbacks=callbacks
        )

        if start_epoch == 0:
            self.storage.reset(self.model_config.render_configuration())
            training_info.initialize()
            reinforcer.initialize_training(training_info)
        else:
            model_state, hidden_state = self.storage.load(training_info)
            reinforcer.initialize_training(training_info, model_state, hidden_state)

        return training_info

    def _openai_logging(self, epoch_result):
        """ Use OpenAI logging facilities for the same type of logging """
        for key in sorted(epoch_result.keys()):
            if key == 'fps':
                # Not super elegant, but I like nicer display of FPS
                openai_logger.record_tabular(key, int(epoch_result[key]))
            else:
                openai_logger.record_tabular(key, epoch_result[key])

        openai_logger.dump_tabular()


def create(model_config, reinforcer, optimizer, storage, total_frames, batches_per_epoch,
           callbacks=None, scheduler=None, openai_logging=False):
    """ Vel factory function """
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
        openai_logging=openai_logging
    )
