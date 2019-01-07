from .info import EpochInfo, BatchInfo, TrainingInfo


class Callback:
    """
    An abstract class that all callback classes extends from.
    """

    def on_initialization(self, training_info: TrainingInfo) -> None:
        """
        Runs for the first time a training process is started from scratch. Is guaranteed to be run only once
        for the training process. Will be run before all other callbacks.
        """
        pass

    def on_train_begin(self, training_info: TrainingInfo) -> None:
        """
        Beginning of a training process - is run every time a training process is started, even if it's restarted from
        a checkpoint.
        """
        pass

    def on_train_end(self, training_info: TrainingInfo) -> None:
        """
        Finalize training process. Runs each time at the end of a training process.
        """
        pass

    def on_epoch_begin(self, epoch_info: EpochInfo) -> None:
        """
        Run for each epoch before an epoch will be trained
        """
        pass

    def on_epoch_end(self, epoch_info: EpochInfo) -> None:
        """
        Runs for each epoch after an epoch is trained
        """
        pass

    def on_batch_begin(self, batch_info: BatchInfo) -> None:
        """
        Runs for each batch before batch is evaluated
        """
        pass

    def on_batch_end(self, batch_info: BatchInfo) -> None:
        """
        Runs for each batch after batch is evaluated
        """
        pass

    def on_validation_batch_begin(self, batch_info: BatchInfo) -> None:
        """
        Supervised learning only - runs before validation batch
        """
        pass

    def on_validation_batch_end(self, batch_info: BatchInfo) -> None:
        """
        Supervised learning only - runs after validation batch
        """
        pass

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict) -> None:
        """
        Persist callback state to the state dictionary
        """
        pass

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict) -> None:
        """
        Load callback state from the state dictionary
        """
        pass

    # TODO(jerry): provide some info about current phase
    # def on_phase_begin(self) -> None: pass
    #
    # def on_phase_end(self) -> None: pass
