import numpy as np
import torch

from vel.rl.api.base import ReplayEnvRollerBase, EnvRollerFactory, EnvRollerBase

from vel.rl.buffers.deque_backend import DequeBufferBackend


class DequeReplayQRoller(ReplayEnvRollerBase):
    """
    Environment roller for action-value models using experience replay.
    Simplest buffer implementation just holding up to given number of samples.

    Because framestack is implemented directly in the buffer, we can use *much* less space to hold samples in
    memory for very little additional cost.
    """
    def __init__(self, environment, device, buffer_capacity: int, buffer_initial_size: int, frame_stack: int):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack

        self.device = device
        self.environment = environment
        self.backend = DequeBufferBackend(
            buffer_capacity=self.buffer_capacity,
            observation_space=environment.observation_space,
            action_space=environment.action_space
        )

        self.last_observation = environment.reset()

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
        indexes = self.backend.sample_batch_uniform(batch_size, self.frame_stack)

        batch = self.backend.get_batch(indexes, self.frame_stack)
        batch['weights'] = np.ones_like(batch['rewards'])
        return batch


class DequeReplayQRollerFactory(EnvRollerFactory):
    """ Factory class for DequeReplayQRoller """
    def __init__(self, buffer_capacity: int, buffer_initial_size: int, frame_stack: int=1):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack = frame_stack

    def instantiate(self, environment, device, settings) -> EnvRollerBase:
        return DequeReplayQRoller(
            environment, device, self.buffer_capacity, self.buffer_initial_size, self.frame_stack
        )


def create(buffer_capacity: int, buffer_initial_size: int, frame_stack: int=1):
    return DequeReplayQRollerFactory(
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        frame_stack=frame_stack
    )
