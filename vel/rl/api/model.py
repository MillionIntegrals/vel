import torch

from vel.api import Model

from .rollout import Rollout
from .evaluator import Evaluator


class RlModel(Model):
    """ Reinforcement learning model """

    def step(self, observations) -> dict:
        """
        Evaluate environment on given observations, return actions and potentially some extra information
        in a dictionary.
        """
        raise NotImplementedError

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        raise NotImplementedError


class RlRnnModel(Model):
    """ Reinforcement learning recurrent model """

    @property
    def is_recurrent(self) -> bool:
        """ If the network is recurrent and needs to be fed previous state """
        return True

    def step(self, observations, state) -> dict:
        """
        Evaluate environment on given observations, return actions and potentially some extra information
        in a dictionary.
        """
        raise NotImplementedError

    @property
    def state_dim(self) -> int:
        """ Dimension of model state """
        raise NotImplementedError

    def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, self.state_dim)

    def evaluate(self, rollout: Rollout) -> Evaluator:
        """ Evaluate model on a rollout """
        raise NotImplementedError
