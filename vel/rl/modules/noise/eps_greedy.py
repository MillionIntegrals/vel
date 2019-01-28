import typing

import torch
import torch.nn as nn

from vel.api import Schedule
from vel.internals.generic_factory import GenericFactory
from vel.schedules.constant import ConstantSchedule


class EpsGreedy(nn.Module):
    """ Epsilon-greedy action selection """
    def __init__(self, epsilon: typing.Union[Schedule, float], environment):
        super().__init__()

        if isinstance(epsilon, Schedule):
            self.epsilon_schedule = epsilon
        else:
            self.epsilon_schedule = ConstantSchedule(epsilon)

        self.action_space = environment.action_space

    def forward(self, actions, batch_info=None):
        if batch_info is None:
            # Just take final value if there is no batch info
            epsilon = self.epsilon_schedule.value(1.0)
        else:
            epsilon = self.epsilon_schedule.value(batch_info['progress'])

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
