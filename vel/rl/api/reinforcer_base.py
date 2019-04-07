import torch

from vel.api import TrainingInfo, EpochInfo, BatchInfo, Model


class ReinforcerBase:
    """
    Manages training process of a single model.
    Learner version for reinforcement-learning problems.
    """

    def initialize_training(self, training_info: TrainingInfo, model_state=None, hidden_state=None):
        """ Run the initialization procedure """
        pass

    def train_epoch(self, epoch_info: EpochInfo):
        """ Train model on an epoch of a fixed number of batch updates """
        raise NotImplementedError

    def train_batch(self, batch_info: BatchInfo):
        """ Single, most atomic 'step' of learning this reinforcer can perform """
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
