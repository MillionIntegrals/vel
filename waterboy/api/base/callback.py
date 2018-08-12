from ..info import EpochInfo, BatchInfo, TrainingInfo


class Callback:
    """
    An abstract class that all callback classes extends from.
    """

    def on_train_begin(self, training_info: TrainingInfo) -> None: pass

    def on_train_end(self, training_info: TrainingInfo) -> None: pass

    def on_epoch_begin(self, epoch_info: EpochInfo) -> None: pass

    def on_epoch_end(self, epoch_info: EpochInfo) -> None: pass

    def on_batch_begin(self, batch_info: BatchInfo) -> None: pass

    def on_batch_end(self, batch_info: BatchInfo) -> None: pass

    def on_validation_batch_begin(self, batch_info: BatchInfo) -> None: pass

    def on_validation_batch_end(self, batch_info: BatchInfo) -> None: pass

    def write_state_dict(self, hidden_state_dict: dict) -> None: pass

    def load_state_dict(self, hidden_state_dict: dict) -> None: pass

    # TODO(jerry): provide some info about current phase
    # def on_phase_begin(self) -> None: pass
    #
    # def on_phase_end(self) -> None: pass
