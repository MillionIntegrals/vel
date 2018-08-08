import torch
import numpy as np
import bisect

from waterboy.api import Learner, EpochIdx
from waterboy.api.metrics import TrainingHistory


class PhaseTrainCommand:
    """ Training  command - learn according to a set of phases """

    def __init__(self, model_config, model_factory, source, storage, phases, callbacks=None, restart=True):
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

        callbacks = []

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        metrics = learner.metrics()

        if self.model_config.reset:
            last_epoch, hidden_state = 0, {}
        else:
            last_epoch, hidden_state = self.storage.resume_learning(learner.model)

        current_phase_idx = self._select_phase_left_bound(last_epoch)
        current_phase = self.phases[current_phase_idx]
        local_idx = last_epoch - self.ladder[current_phase_idx]

        current_phase.set_up_phase(learner, self.source, metrics, callbacks)
        print(current_phase.banner())

        if last_epoch > 0:
            current_phase.restore(EpochIdx(last_epoch, local_idx), learner, hidden_state)

        for callback in callbacks:
            callback.on_train_begin()

        training_history = TrainingHistory()

        for global_epoch_idx in range(last_epoch + 1, self.full_number_of_epochs+1):
            iteration_phase_idx = self._select_phase_right_bound(global_epoch_idx-1)
            local_idx = global_epoch_idx - self.ladder[iteration_phase_idx]

            epoch_idx = EpochIdx(global_epoch_idx, local_idx)

            # Phase preparations
            while current_phase_idx != iteration_phase_idx:
                current_phase.tear_down_phase(learner)

                current_phase_idx += 1
                current_phase = self.phases[current_phase_idx]

                current_phase.set_up_phase(learner, self.source, metrics, callbacks)
                print(current_phase.banner())

            # Main training piece
            epoch_result = current_phase.execute_epoch(epoch_idx, learner)

            self.storage.checkpoint(
                global_epoch_idx, epoch_result, learner.model, state_dict=current_phase.state_dict()
            )

            training_history.add(epoch_result)

        for callback in callbacks:
            callback.on_train_end()

        # Tear down the last phase
        if current_phase is not None:
            current_phase.tear_down_phase(learner)


def create(model_config, model, source, storage, phases, callbacks=None, restart=True):
    """ Waterboy creation function """
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
