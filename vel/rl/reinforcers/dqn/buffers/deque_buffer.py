import torch
import gym
import numpy as np

from vel.api.base import Model
from ..dqn_reinforcer import DqnBufferBase
from vel.rl.buffers.deque_backend import DequeBufferBackend


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
        indexes = self.backend.sample_batch_uniform(batch_size, self.frame_stack)
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
