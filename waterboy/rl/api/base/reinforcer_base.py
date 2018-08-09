import torch
from torch.optim import Optimizer

from waterboy.api.base import Model
from waterboy.api.metrics import EpochResultAccumulator
from waterboy.api import EpochIdx, BatchIdx


class ReinforcerBase:
    """
    Manages training process of a single model.
    Learner version for reinforcement-learning problems.
    """
    def initialize_training(self) -> None:
        """ Prepare all variables for training from scratch """
        pass

    def train_step(self, batch_idx: BatchIdx, optimizer: Optimizer,
                   result_accumulator: EpochResultAccumulator=None) -> None:
        """ Single, most atomic 'step' of learning this reinforcer can perform """
        raise NotImplementedError

    def train_epoch(self, epoch_idx: EpochIdx, batches_per_epoch: int, optimizer: Optimizer,
                    callbacks: list, result_accumulator: EpochResultAccumulator=None) -> dict:
        """ Train model on an epoch of a fixed number of batch updates """
        raise NotImplementedError

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        raise NotImplementedError

    @property
    def model(self) -> Model:
        """ Model trained by this reinforcer """
        raise NotImplementedError


class ReinforcerFactory:
    """ A reinforcer factory """
    def instantiate(self, device: torch.device) -> ReinforcerBase:
        """ Create new reinforcer instance """
        raise NotImplementedError
