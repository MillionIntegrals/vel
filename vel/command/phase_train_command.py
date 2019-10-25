import numpy as np
import bisect
import typing

import vel.api as api
import vel.train as train

from vel.data.loader import DatasetLoader
from vel.metric.samples_per_sec import SamplesPerSec
from vel.callback.time_tracker import TimeTracker
from vel.callback.sample_tracker import SampleTracker


class PhaseTrainCommand:
    """ Training  command - learn according to a set of phases """

    def __init__(self, model_config: api.ModelConfig, model_factory: api.ModuleFactory, loader: DatasetLoader,
                 storage: api.Storage, phases: typing.List[train.TrainPhase],
                 callbacks=None, restart=True):
        self.model_config = model_config
        self.model_factory = model_factory
        self.loader = loader
        self.storage = storage
        self.phases = phases
        self.ladder = self._build_phase_ladder(phases)
        self.full_number_of_epochs = sum(p.number_of_epochs for p in phases)
        self.callbacks = callbacks if callbacks is not None else []
        self.restart = restart

    @staticmethod
    def _build_phase_ladder(phases):
        """ Build a ladder of learning phases """
        return [0] + np.cumsum([p.number_of_epochs for p in phases]).tolist()[:-1]

    def _select_phase_left_bound(self, epoch_number):
        """
        Return number of current phase.
        Return index of first phase not done after all up to epoch_number were done.
        """
        idx = bisect.bisect_left(self.ladder, epoch_number)

        if idx >= len(self.ladder):
            return len(self.ladder) - 1
        elif self.ladder[idx] > epoch_number:
            return idx - 1
        else:
            return idx

    def _select_phase_right_bound(self, epoch_number):
        """
        Return number of current phase.
        Return index of first phase not done after all up to epoch_number were done.
        """
        return bisect.bisect_right(self.ladder, epoch_number) - 1

    def run(self):
        """ Run the command with supplied configuration """
        device = self.model_config.torch_device()
        trainer = train.Trainer(device, self.model_factory.instantiate())

        # Check if training was already started and potentially continue where we left off
        training_info, hidden_state = self.start_training(trainer)

        # Prepare current training phase
        current_phase_idx = self._select_phase_left_bound(training_info.start_epoch_idx)
        current_phase = self.phases[current_phase_idx]
        local_idx = training_info.start_epoch_idx - self.ladder[current_phase_idx]

        current_phase.set_up_phase(training_info, trainer.model, self.loader)
        print(current_phase.banner())

        if training_info.start_epoch_idx > 0:
            current_phase.restore(training_info, local_idx, trainer.model, hidden_state)

        training_info.on_train_begin(trainer.model)

        for global_epoch_idx in range(training_info.start_epoch_idx + 1, self.full_number_of_epochs + 1):
            iteration_phase_idx = self._select_phase_right_bound(global_epoch_idx-1)
            local_idx = global_epoch_idx - self.ladder[iteration_phase_idx]

            # Phase preparations
            while current_phase_idx != iteration_phase_idx:
                current_phase.tear_down_phase(training_info, trainer.model)

                current_phase_idx += 1
                current_phase = self.phases[current_phase_idx]

                current_phase.set_up_phase(training_info, trainer.model, self.loader)
                print(current_phase.banner())

            # Create epoch info
            epoch_info = current_phase.epoch_info(training_info, global_epoch_idx, local_idx)

            # Execute learning
            current_phase.execute_epoch(epoch_info, trainer)

            # Epoch checkpoint
            self.storage.checkpoint(epoch_info, trainer.model)

        # Tear down the last phase
        if current_phase is not None:
            current_phase.tear_down_phase(training_info, trainer.model)

        training_info.on_train_end()

        return training_info

    def start_training(self, trainer) -> (api.TrainingInfo, dict):
        """ Possibly resume training from a saved state from the storage """
        if self.model_config.resume_training:
            start_epoch = self.storage.last_epoch_idx()
        else:
            start_epoch = 0

        # Initial set of callbacks, always useful
        callbacks = [TimeTracker(), SampleTracker()]

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        # Metrics to track through this training
        metrics = trainer.metrics() + [SamplesPerSec()]

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
            hidden_state = None
        else:
            model_state, hidden_state = self.storage.load(training_info)
            training_info.restore(hidden_state)

            trainer.initialize_training(training_info, model_state, hidden_state)

        return training_info, hidden_state


def create(model_config, model, loader, storage, phases, callbacks=None, restart=True):
    """ Vel factory function """
    return PhaseTrainCommand(
        model_config=model_config,
        model_factory=model,
        loader=loader,
        storage=storage,
        phases=phases,
        callbacks=callbacks,
        restart=restart
    )
