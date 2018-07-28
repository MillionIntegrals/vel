import torch
from torch.optim import Optimizer

from waterboy.api import BatchIdx
from waterboy.api.base import Model
from waterboy.api.metrics import EpochResultAccumulator


class ReinforcerBase:
    def train_batch(self, batch_idx: BatchIdx, optimizer: Optimizer, result_accumulator: EpochResultAccumulator=None) -> None:
        raise NotImplementedError


class ReinforcerFactoryBase:
    def instantiate(self, device: torch.device, model: Model) -> ReinforcerBase:
        raise NotImplementedError
