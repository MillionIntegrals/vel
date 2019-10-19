import gym
import typing

import torch

from vel.api import Schedule, VModule
from vel.internal.generic_factory import GenericFactory


class EpsGreedy(VModule):
    """ Epsilon-greedy action selection """
    def __init__(self, action_space: gym.Space):
        super().__init__()

        self.action_space = action_space

    def forward(self, actions, epsilon, deterministic=False):
        if deterministic:
            return actions
        else:
            random_samples = torch.randint_like(actions, self.action_space.n)
            selector = torch.rand_like(random_samples, dtype=torch.float32)

            # Actions with noise applied
            noisy_actions = torch.where(selector > epsilon, actions, random_samples)

            return noisy_actions

    def reset_training_state(self, dones, batch_info):
        """ A hook for a model to react when during training episode is finished """
        pass


def create(epsilon: typing.Union[Schedule, float]):
    """ Vel factory function """
    return GenericFactory(EpsGreedy, arguments={'epsilon': epsilon})
