import numpy as np
import torch

from vel.api.base import Schedule
from vel.rl.api.base import ReplayEnvRollerBase, EnvRollerFactory
from vel.rl.buffers.prioritized_backend import PrioritizedReplayBackend


class PrioritizedReplayQRoller(ReplayEnvRollerBase):
    """
    Environment roller for action-value models using experience replay.
    Experience replay buffer implementation with prioritized sampling based on td-errors

    Because framestack is implemented directly in the buffer, we can use *much* less space to hold samples in
    memory for very little additional cost.
    """

    def __init__(self, environment, device, buffer_capacity: int, buffer_initial_size: int, frame_stack: int,
                 priority_exponent: float, priority_weight: Schedule, priority_epsilon: float):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack

        self.priority_exponent = priority_exponent
        self.priority_weight_schedule = priority_weight
        self.priority_epsilon = priority_epsilon

        self.environment = environment
        self.device = device
        self.last_observation = environment.reset()

        self.backend = PrioritizedReplayBackend(
            buffer_capacity=self.buffer_capacity,
            observation_space=environment.observation_space,
            action_space=environment.action_space
        )

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        return self.backend.current_size >= self.buffer_initial_size

    def rollout(self, batch_info, model) -> dict:
        """ Roll-out the environment and return it """
        epsilon_value = batch_info['epsilon_value']

        last_observation = np.concatenate([
            self.backend.get_frame(self.backend.current_idx, self.frame_stack - 1),
            self.last_observation
        ], axis=-1)

        observation_tensor = torch.from_numpy(last_observation[None]).to(self.device)
        action = model.step(observation_tensor, epsilon_value)['actions'].item()

        observation, reward, done, info = self.environment.step(action)

        self.backend.store_transition(self.last_observation, action, reward, done)

        # Usual, reset on done
        if done:
            observation = self.environment.reset()

        self.last_observation = observation

        return info.get('episode')

    def sample(self, batch_info, batch_size, model) -> dict:
        """ Sample experience from replay buffer and return a batch """
        probs, indexes, tree_idxs = self.backend.sample_batch_prioritized(batch_size, self.frame_stack)
        batch = self.backend.get_batch(indexes, self.frame_stack)

        # Normalize weights properly
        priority_weight = self.priority_weight_schedule.value(batch_info['progress'])

        probs = np.stack(probs) / self.backend.segment_tree.total()
        capacity = self.backend.deque.current_size
        weights = (capacity * probs) ** (-priority_weight)
        weights = weights / weights.max()

        batch['weights'] = weights
        batch['tree_idxs'] = tree_idxs

        return batch

    def update(self, sample, errors):
        """ Update sample weights in the priority buffer """
        weights = (errors + self.priority_epsilon) ** self.priority_exponent

        for idx, priority in zip(sample['tree_idxs'], weights):
            self.backend.update_priority(idx, priority)


class PrioritizedReplayQRollerFactory(EnvRollerFactory):
    """ Factory class for PrioritizedReplayQRoller """

    def __init__(self, buffer_capacity: int, buffer_initial_size: int, frame_stack: int, priority_exponent: float,
                 priority_weight: Schedule, priority_epsilon: float):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack
        self.priority_exponent = priority_exponent
        self.priority_weight = priority_weight
        self.priority_epsilon = priority_epsilon

    def instantiate(self, environment, device, settings):
        return PrioritizedReplayQRoller(
            environment=environment,
            device=device,
            buffer_capacity=self.buffer_capacity,
            buffer_initial_size=self.buffer_initial_size,
            frame_stack=self.frame_stack,
            priority_exponent=self.priority_exponent,
            priority_weight=self.priority_weight,
            priority_epsilon=self.priority_epsilon
        )


def create(buffer_capacity: int, buffer_initial_size: int, frame_stack: int, priority_exponent: float,
           priority_weight: Schedule, priority_epsilon: float):
    return PrioritizedReplayQRollerFactory(
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        frame_stack=frame_stack,
        priority_exponent=priority_exponent,
        priority_weight=priority_weight,
        priority_epsilon=priority_epsilon
    )
