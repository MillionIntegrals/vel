import torch
import gym
import numpy as np

from vel.api.base import Model
from vel.exceptions import VelException
from ..dqn_reinforcer import DqnBufferBase


class DequeBufferBackend:
    """ Simple backend behind DequeBuffer """

    def __init__(self, buffer_capacity: int, frame_shape):
        # Maximum number of items in the buffer
        self.buffer_capacity = buffer_capacity

        # How many elements have been inserted in the buffer
        self.current_size = 0

        # Index of last inserted element
        self.current_idx = -1

        # Data buffers
        self.frame_buffer = np.zeros([self.buffer_capacity] + list(frame_shape), dtype=np.uint8)
        self.action_buffer = np.zeros([self.buffer_capacity], dtype=np.int)
        self.reward_buffer = np.zeros([self.buffer_capacity], dtype=float)
        self.dones_buffer = np.zeros([self.buffer_capacity], dtype=bool)

        # Just a sentinel to simplify further calculations
        self.dones_buffer[self.current_idx] = True

    def store_transition(self, frame, action, reward, done):
        """ Store given transition in the backend """
        self.current_idx = (self.current_idx + 1) % self.buffer_capacity

        self.frame_buffer[self.current_idx] = frame
        self.action_buffer[self.current_idx] = action
        self.reward_buffer[self.current_idx] = reward
        self.dones_buffer[self.current_idx] = done

        if self.current_size < self.buffer_capacity:
            self.current_size += 1

        return self.current_idx

    def get_frame(self, idx, history):
        """ Return frame from the buffer """
        if idx >= self.current_size:
            raise VelException("Requested frame beyond the size of the buffer")

        accumulator = []

        last_frame = self.frame_buffer[idx]
        accumulator.append(last_frame)

        for i in range(history - 1):
            prev_idx = (idx - 1) % self.buffer_capacity

            if prev_idx == self.current_idx:
                raise VelException("Cannot provide enough history for the frame")
            elif self.dones_buffer[prev_idx]:
                # If previous frame was done - just append zeroes
                accumulator.append(np.zeros_like(last_frame))
            else:
                idx = prev_idx
                accumulator.append(self.frame_buffer[idx])

        # We're pushing the elements in reverse order
        return np.concatenate(accumulator[::-1], axis=-1)

    def get_frame_with_future(self, idx, history):
        """ Return frame from the buffer together with the next frame """
        if idx == self.current_idx:
            raise VelException("Cannot provide enough future for the frame")

        past_frame = self.get_frame(idx, history)

        future_frame = np.zeros_like(past_frame)

        future_frame[:, :, :-1] = past_frame[:, :, 1:]

        if not self.dones_buffer[idx]:
            next_idx = (idx + 1) % self.buffer_capacity
            next_frame = self.frame_buffer[next_idx]
            future_frame[:, :, -1:] = next_frame

        return past_frame, future_frame

    def get_batch(self, indexes, history):
        """ Return batch with given indexes """
        frame_batch_shape = (
                [indexes.shape[0]] + list(self.frame_buffer.shape[1:-1]) + [self.frame_buffer.shape[-1] * history]
        )

        past_frame_buffer = np.zeros(frame_batch_shape, dtype=np.uint8)
        future_frame_buffer = np.zeros(frame_batch_shape, dtype=np.uint8)

        for buffer_idx, frame_idx in enumerate(indexes):
            past_frame_buffer[buffer_idx], future_frame_buffer[buffer_idx] = self.get_frame_with_future(frame_idx, history)

        actions = self.action_buffer[indexes]
        rewards = self.reward_buffer[indexes]
        dones = self.dones_buffer[indexes]

        return past_frame_buffer, actions, rewards, future_frame_buffer, dones

    def uniform_batch_sample(self, batch_size, history):
        """ Return indexes of next sample"""
        # Sample from up to total size
        if self.current_size < self.buffer_capacity:
            # -1 because we cannot take the last one
            return np.random.choice(self.current_size - 1, batch_size, replace=False)
        else:
            candidate = np.random.choice(self.buffer_capacity, batch_size, replace=False)

            forbidden_ones = (
                    np.arange(self.current_idx, self.current_idx + history)
                    % self.buffer_capacity
            )

            # Exclude these frames for learning as they may have some part of history overwritten
            while any(x in candidate for x in forbidden_ones):
                candidate = np.random.choice(self.buffer_capacity, batch_size, replace=False)

            return candidate


class DequeBuffer(DqnBufferBase):
    """
    Simplest buffer just holding up to given number of samples.

    Because framestack is implemented directly in the buffer, we can use *much* less space to hold samples in
    memory for very little additional cost.
    """
    def __init__(self, buffer_capacity: int, buffer_initial_size: int, frame_stack: int):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack

        # Awaiting initialization
        self.last_observation = None
        self.device = None
        self.backend: DequeBufferBackend = None

    def initialize(self, environment: gym.Env, device: torch.device):
        """ Initialze buffer for operation """
        self.device = device
        self.backend = DequeBufferBackend(
            buffer_capacity=self.buffer_capacity,
            frame_shape=environment.observation_space.shape
        )

        self.last_observation = environment.reset()

    def is_ready(self) -> bool:
        """ If buffer is ready for training """
        return self.backend.current_size >= self.buffer_initial_size

    def rollout(self, environment: gym.Env, model: Model, epsilon_value: float) -> dict:
        """ Evaluate model and proceed one step forward with the environment. Store result in the replay buffer """
        # last_observation_idx = self._store_frame(self.last_observation)

        # last_observation = self._get_frame(last_observation_idx)

        last_observation = np.concatenate([
            self.backend.get_frame(self.backend.current_idx, self.frame_stack - 1),
            self.last_observation
        ], axis=-1)

        observation_tensor = torch.from_numpy(last_observation[None]).to(self.device)
        action = model.step(observation_tensor, epsilon_value).item()

        observation, reward, done, info = environment.step(action)

        self.backend.store_transition(self.last_observation, action, reward, done)

        # Usual, reset on done
        if done:
            observation = environment.reset()

        self.last_observation = observation

        return info.get('episode')

    def sample(self, batch_info, batch_size) -> dict:
        """ Calculate random sample from the replay buffer """
        indexes = self.backend.uniform_batch_sample(batch_size, self.frame_stack)
        observations, actions, rewards, observations_tplus1, dones = self.backend.get_batch(indexes, self.frame_stack)

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'observations_tplus1': observations_tplus1,
            'weights': np.ones_like(rewards)
        }


def create(buffer_capacity: int, buffer_initial_size: int, frame_stack: int=1):
    return DequeBuffer(
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        frame_stack=frame_stack
    )
