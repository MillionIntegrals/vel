import torch
import numpy as np
import bisect
import typing

from vel.api import Learner, TrainingInfo
from vel.api.base import TrainPhase


class PhaseTrainCommand:
    """ Training  command - learn according to a set of phases """

    def __init__(self, model_config, model_factory, source, storage, phases: typing.List[TrainPhase],
                 callbacks=None, restart=True):
        self.model_config = model_config
        self.model_factory = model_factory
        self.source = source
        self.storage = storage
        self.phases = phases
        self.ladder = self._build_phase_ladder(phases)
        self.full_number_of_epochs = sum(p.number_of_epochs for p in phases)
        self.callbacks = callbacks
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
        device = torch.device(self.model_config.device)
        learner = Learner(device, self.model_factory.instantiate())

        # All callbacks useful for learning
        callbacks = self.gather_callbacks()

        # Metrics to track through this training
        metrics = learner.metrics()

        # Check if training was already started and potentially continue where we left off
        training_info, hidden_state = self.resume_training(learner, callbacks, metrics)

        # Prepare current training phase
        current_phase_idx = self._select_phase_left_bound(training_info.start_epoch_idx)
        current_phase = self.phases[current_phase_idx]
        local_idx = training_info.start_epoch_idx - self.ladder[current_phase_idx]

        current_phase.set_up_phase(training_info, learner.model, self.source)
        print(current_phase.banner())

        if training_info.start_epoch_idx > 0:
            current_phase.restore(training_info, local_idx, learner.model, hidden_state)

        for callback in callbacks:
            callback.on_train_begin(training_info)

        for global_epoch_idx in range(training_info.start_epoch_idx + 1, self.full_number_of_epochs + 1):
            iteration_phase_idx = self._select_phase_right_bound(global_epoch_idx-1)
            local_idx = global_epoch_idx - self.ladder[iteration_phase_idx]

            # Phase preparations
            while current_phase_idx != iteration_phase_idx:
                current_phase.tear_down_phase(training_info, learner.model)

                current_phase_idx += 1
                current_phase = self.phases[current_phase_idx]

                current_phase.set_up_phase(training_info, learner.model, self.source)
                print(current_phase.banner())

            # Create epoch info
            epoch_info = current_phase.epoch_info(training_info, global_epoch_idx, local_idx)

            # Execute learning
            current_phase.execute_epoch(epoch_info, learner)

            # Epoch checkpoint
            self.storage.checkpoint(epoch_info, learner.model)

            # Gather results
            training_info.history.add(epoch_info.result)

        for callback in callbacks:
            callback.on_train_end(training_info)

        # Tear down the last phase
        if current_phase is not None:
            current_phase.tear_down_phase(training_info, learner.model)

        return training_info

    def gather_callbacks(self) -> list:
        """ Gather all the callbacks to be used in this training run """
        callbacks = []

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        return callbacks

    def resume_training(self, learner, callbacks, metrics) -> (TrainingInfo, dict):
        """ Possibly resume training from a saved state from the storage """
        if self.model_config.reset:
            start_epoch, hidden_state = 0, {}
        else:
            start_epoch, hidden_state = self.storage.resume_learning(learner.model)

        training_info = TrainingInfo(start_epoch_idx=start_epoch, metrics=metrics, callbacks=callbacks)

        if start_epoch > 0:
            for callback in callbacks:
                callback.load_state_dict(hidden_state)

            training_info.restore(hidden_state)

        return training_info, hidden_state


def create(model_config, model, source, storage, phases, callbacks=None, restart=True):
    """ Vel creation function """
    callbacks = callbacks or []
    return PhaseTrainCommand(
        model_config=model_config,
        model_factory=model,
        source=source,
        storage=storage,
        phases=phases,
        callbacks=callbacks,
        restart=restart
    )
