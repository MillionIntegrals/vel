import torch.optim

from ..progress_idx import EpochIdx, BatchIdx


class Callback:
    """
    An abstract class that all callback classes extends from.
    """

    def on_train_begin(self) -> None: pass

    def on_train_end(self) -> None: pass

    def on_phase_begin(self) -> None: pass

    def on_phase_end(self) -> None: pass

    def on_epoch_begin(self, epoch_idx: EpochIdx) -> None: pass

    def on_epoch_end(self, epoch_idx: EpochIdx, metrics: dict) -> None: pass

    def on_batch_begin(self, batch_idx: BatchIdx) -> None: pass

    def on_batch_end(self, batch_idx: BatchIdx, metrics: dict, optimizer: torch.optim.Optimizer) -> None: pass

    def on_validation_begin(self, epoch_idx: EpochIdx) -> None: pass

    def on_validation_end(self, epoch_idx: EpochIdx, metrics: dict) -> None: pass

    def write_state_dict(self, hidden_state_dict: dict) -> None: pass

    def load_state_dict(self, hidden_state_dict: dict) -> None: pass
