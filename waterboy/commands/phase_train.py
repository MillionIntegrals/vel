import torch
import numpy as np
import bisect

from waterboy.api import Learner
from waterboy.api.metrics import TrainingHistory


class PhaseTrainCommand:
    """ Training  command - learn according to a set of phases """

    def __init__(self, model_config, model, source, storage, phases, callbacks=None):
        self.model_config = model_config
        self.model = model
        self.source = source
        self.storage = storage
        self.phases = phases
        self.ladder = self._build_phase_ladder(phases)
        self.full_number_of_epochs = sum(p.number_of_epochs for p in phases)
        self.callbacks = callbacks

    @staticmethod
    def _build_phase_ladder(phases):
        """ Build a ladder of learning phases """
        return [0] + np.cumsum([p.number_of_epochs for p in phases]).tolist()[:-1]

    def run(self):
        """ Run the command with supplied configuration """
        device = torch.device(self.model_config.device)
        learner = Learner(device, self.model)

        callbacks = []

        callbacks.extend(self.callbacks)
        callbacks.extend(self.storage.streaming_callbacks())

        metrics = learner.metrics()

        print("-" * 120)
        learner.summary()
        print("-" * 120)
        print("Number of model parameters: {:,}".format(learner.number_of_parameters()))
        print("-" * 120)

        last_epoch, hidden_state = self.storage.resume_learning(learner.model)

        current_idx = bisect.bisect_right(self.ladder, last_epoch) - 1
        current_phase = self.phases[current_idx]
        local_idx = last_epoch - self.ladder[current_idx]

        current_phase.set_up_phase(learner, self.source, metrics, callbacks)

        if last_epoch > 0:
            current_phase.restore(local_idx, learner, hidden_state)

        for callback in callbacks:
            callback.on_train_begin()

        training_history = TrainingHistory()

        for epoch_idx in range(1 + last_epoch, self.full_number_of_epochs+1):
            phase_idx = bisect.bisect_right(self.ladder, last_epoch) - 1
            local_idx = last_epoch - self.ladder[current_idx]

            # Phase preparations
            if current_idx != phase_idx:
                current_phase.tear_down_phase(learner)

                current_idx = phase_idx
                current_phase = self.phases[current_idx]

                current_phase.set_up_phase(learner, self.source, metrics, callbacks)

            # Main training piece
            epoch_result = current_phase.execute_epoch(local_idx, learner)

            self.storage.checkpoint(epoch_idx, epoch_result, learner.model, state_dict=current_phase.state_dict())

            training_history.add(epoch_result)

        for callback in callbacks:
            callback.on_train_end()

        # Tear down the last phase
        if current_phase is not None:
            current_phase.tear_down_phase(learner)


def create(model_config, model, source, storage, phases, callbacks=None):
    """ Waterboy creation function """
    callbacks = callbacks or []
    return PhaseTrainCommand(
        model_config=model_config,
        model=model,
        source=source,
        storage=storage,
        phases=phases,
        callbacks=callbacks
    )

