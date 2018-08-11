import torch
import gym
import numpy as np

from waterboy.api.base import Model
from ..dqn_reinforcer import DqnBufferBase


class DequeBuffer(DqnBufferBase):
    """
    Simplest buffer just holding up to given number of samples.

    Because framestack is implemented directly in the buffer, we can use *much* less space to hold samples in
    memory for very little additional cost.

    Potentially could also compress frames and frames on t+1, but that would complicate the code which I tried
    not to do.
    """
    def __init__(self, buffer_capacity: int, buffer_initial_size: int, frame_stack: int):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack

        # Awaiting initialization
        self.frame_buffer = None
        self.next_frame_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.dones_buffer = None
        self.last_observation = None
        self.device = None

        self.current_idx = -1
        self.total_size = 0

    def initialize(self, environment: gym.Env, device: torch.device):
        """ Initialze buffer for operation """
        self.device = device

        self.frame_buffer = np.zeros([self.buffer_capacity] + list(environment.observation_space.shape), dtype=np.uint8)
        self.next_frame_buffer = np.zeros([self.buffer_capacity] + list(environment.observation_space.shape), dtype=np.uint8)
        self.action_buffer = np.zeros([self.buffer_capacity], dtype=np.int)
        self.reward_buffer = np.zeros([self.buffer_capacity], dtype=float)
        self.dones_buffer = np.zeros([self.buffer_capacity], dtype=bool)
        self.current_idx = -1

        # Just a sentinel to simplify further calculations
        self.dones_buffer[self.current_idx] = True

        self.last_observation = environment.reset()

    def is_ready(self):
        """ If buffer is ready for training """
        return self.total_size >= self.buffer_initial_size

    def rollout(self, environment: gym.Env, model: Model, epsilon_value: float):
        """ Evaluate model and proceed one step forward with the environment. Store result in the replay buffer """
        last_observation_idx = self._store_frame(self.last_observation)

        last_observation = self._get_frame(last_observation_idx)
        observation_tensor = torch.from_numpy(last_observation[None]).to(self.device)
        action = model.step(observation_tensor, epsilon_value).item()

        observation, reward, done, info = environment.step(action)
        observation = observation[:]

        self._store_transition(observation, action, reward, done)

        # Usual, reset on done
        if done:
            observation = environment.reset()[:]

        self.last_observation = observation[:]

        return info.get('episode')

    def sample(self, batch_size) -> dict:
        """ Calculate random sample from the replay buffer """
        indexes = self._sample_indexes(batch_size)

        # States, states t+1
        observations, observations_tplus1 = self._unpack_states(indexes)
        # Actions
        actions = self.action_buffer[indexes]
        # Rewards
        rewards = self.reward_buffer[indexes]
        # Dones
        dones = self.dones_buffer[indexes]

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'observations_tplus1': observations_tplus1
        }

    def _store_frame(self, frame):
        """ Add another frame to the buffer """
        self.current_idx = (self.current_idx + 1) % self.buffer_capacity
        self.frame_buffer[self.current_idx] = frame

        if self.total_size < self.buffer_capacity:
            self.total_size += 1

        return self.current_idx

    def _get_frame(self, idx):
        """ Return frame from the buffer """
        accumulator = []

        last_frame = self.frame_buffer[idx]
        accumulator.append(last_frame)

        for i in range(self.frame_stack - 1):
            prev_idx = idx - 1

            if self.dones_buffer[prev_idx]:
                # If previous frame was done -
                accumulator.append(np.zeros_like(last_frame))
            else:
                idx = prev_idx
                accumulator.append(self.frame_buffer[idx])

        # We're pushing the elements in reverse order
        return np.concatenate(accumulator[::-1], axis=-1)

    def _store_transition(self, next_observation, action, reward, done):
        """ Add frame transition to the buffer """
        self.next_frame_buffer[self.current_idx] = next_observation
        self.action_buffer[self.current_idx] = action
        self.reward_buffer[self.current_idx] = reward
        self.dones_buffer[self.current_idx] = done

    def _unpack_states(self, indexes):
        """ Unpack states from frame buffer into observation arrays """
        observation_shape = (
                [indexes.shape[0]] + list(self.frame_buffer.shape[1:-1]) +
                [self.frame_buffer.shape[-1] * self.frame_stack]
        )

        observations = np.zeros(observation_shape)
        observations_tplus1 = np.zeros_like(observations)

        for idx, frame_idx in enumerate(indexes):
            current_frame = self._get_frame(frame_idx)
            next_frame = np.concatenate([current_frame[:, :, 1:], self.next_frame_buffer[frame_idx]], axis=-1)

            observations[idx] = current_frame
            observations_tplus1[idx] = next_frame

        return observations, observations_tplus1

    def _sample_indexes(self, batch_size):
        """ Return indexes of next sample"""
        # Sample from up to total size
        return np.random.choice(self.total_size, batch_size, replace=False)


def create(buffer_capacity: int, buffer_initial_size: int, frame_stack: int=1):
    return DequeBuffer(
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        frame_stack=frame_stack
    )
