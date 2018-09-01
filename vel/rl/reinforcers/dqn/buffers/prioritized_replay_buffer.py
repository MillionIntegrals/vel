import gym
import numpy as np
import torch


from vel.api.base import Schedule, Model
from vel.rl.buffers.prioritized_backend import PrioritizedReplayBackend
from ..dqn_reinforcer import DqnBufferBase


class PrioritizedReplayBuffer(DqnBufferBase):
    """
    Experience replay buffer with prioritized sampling based on td-errors
    """

    def __init__(self, buffer_capacity: int, buffer_initial_size: int, frame_stack: int, priority_exponent: float,
                 priority_weight: Schedule, epsilon: float):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack

        self.priority_exponent = priority_exponent
        self.priority_weight_schedule = priority_weight
        self.epsilon = epsilon

        # Awaiting initialization
        self.backend: PrioritizedReplayBackend = None
        self.device = None
        self.last_observation = None

    def initialize(self, environment: gym.Env, device: torch.device):
        """ Initialze buffer for operation """
        self.device = device
        self.last_observation = environment.reset()

        self.backend = PrioritizedReplayBackend(
            buffer_capacity=self.buffer_capacity,
            observation_space=environment.observation_space,
            action_space=environment.action_space
        )

    def is_ready(self) -> bool:
        """ If buffer is ready for training """
        return self.backend.current_size >= self.buffer_initial_size

    def rollout(self, environment: gym.Env, model: Model, epsilon_value: float) -> dict:
        """ Evaluate model and proceed one step forward with the environment. Store result in the replay buffer """
        last_observation = np.concatenate([
            self.backend.get_frame(self.backend.current_idx, self.frame_stack - 1),
            self.last_observation
        ], axis=-1)

        observation_tensor = torch.from_numpy(last_observation[None]).to(self.device)
        action = model.step(observation_tensor, epsilon_value)['actions'].item()

        observation, reward, done, info = environment.step(action)

        self.backend.store_transition(self.last_observation, action, reward, done)

        # Usual, reset on done
        if done:
            observation = environment.reset()

        self.last_observation = observation

        return info.get('episode')

    def sample(self, batch_info, batch_size) -> dict:
        """ Calculate random sample from the replay buffer """
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
        weights = errors ** self.priority_exponent

        for idx, priority in zip(sample['tree_idxs'], weights):
            self.backend.update_priority(idx, priority)


def create(buffer_capacity: int, buffer_initial_size: int, frame_stack: int, priority_exponent: float,
           priority_weight: Schedule, epsilon: float):
    return PrioritizedReplayBuffer(
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        frame_stack=frame_stack,
        priority_exponent=priority_exponent,
        priority_weight=priority_weight,
        epsilon=epsilon
    )
