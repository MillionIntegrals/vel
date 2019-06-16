import torch
from vel.api import Model


class Policy(Model):
    """ Base class for reinforcement learning policies """

    def act(self, observation, state=None, deterministic=False) -> dict:
        """ Make an action based on the observation from the environment. """
        raise NotImplementedError

    def value(self, observation, state=None) -> torch.tensor:
        """ Return the expected reward from current state """
        return self.act(observation=observation, state=state)['value']

    def reset_state(self, state, dones):
        """ Reset the state after the episode has been terminated """
        raise NotImplementedError

    def evaluate(self, rollout) -> object:
        """ Return an evaluator object evaluating given rollout that may be used for gradient computations etc. """
        raise NotImplementedError
