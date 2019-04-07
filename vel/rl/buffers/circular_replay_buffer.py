import gym
import torch
import typing

from vel.rl.api import ReplayBuffer, ReplayBufferFactory, Transitions, Trajectories
from .backend.circular_vec_buffer_backend import CircularVecEnvBufferBackend


class CircularReplayBuffer(ReplayBuffer):
    """
    Replay buffer that uses a circular buffer - new experience overwrites the oldest one
    Version supporting multiple environments.

    Frame stack compensation - if environment has a framestack built in, we will store only the last frame
    """

    def __init__(self, buffer_capacity: int, buffer_initial_size: int, num_envs: int, observation_space: gym.Space,
                 action_space: gym.Space, frame_stack_compensation: bool = False, frame_history: int = 1):
        super().__init__()

        self.buffer_initial_size = buffer_initial_size

        self.backend = CircularVecEnvBufferBackend(
            buffer_capacity=buffer_capacity,
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
            frame_stack_compensation=frame_stack_compensation,
            frame_history=frame_history
        )

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        return self.buffer_initial_size <= self.backend.current_size

    def initial_memory_size_hint(self) -> typing.Optional[int]:
        """ Hint how much data is needed to begin sampling, required only for diagnostics """
        return self.buffer_initial_size

    def _get_transitions(self, indexes):
        """ Return batch with given indexes """
        transition_tensors = self.backend.get_transitions(indexes)

        return Trajectories(
            num_steps=indexes.shape[0],
            num_envs=indexes.shape[1],
            environment_information=None,
            transition_tensors=transition_tensors,
            rollout_tensors={}
        ).to_transitions()

    def sample_trajectories(self, rollout_length, batch_info) -> Trajectories:
        """ Sample batch of trajectories and return them """
        indexes = self.backend.sample_batch_trajectories(rollout_length)
        transition_tensors = self.backend.get_trajectories(indexes, rollout_length)

        return Trajectories(
            num_steps=rollout_length,
            num_envs=self.backend.num_envs,
            environment_information=None,
            transition_tensors={k: torch.from_numpy(v) for k, v in transition_tensors.items()},
            rollout_tensors={}
        )

    def sample_transitions(self, batch_size, batch_info, discount_factor=None) -> Transitions:
        """ Sample batch of transitions and return them """
        indexes = self.backend.sample_batch_transitions(batch_size)
        transition_tensors = self.backend.get_transitions(indexes)

        return Trajectories(
            num_steps=batch_size,
            num_envs=self.backend.num_envs,
            environment_information=None,
            transition_tensors={k: torch.from_numpy(v) for k, v in transition_tensors.items()},
            rollout_tensors={}
        ).to_transitions()

    def sample_forward_transitions(self, batch_size, batch_info, forward_steps: int,
                                   discount_factor: float) -> Transitions:
        """
        Sample transitions from replay buffer with _forward steps_.
        That is, instead of getting a transition s_t -> s_t+1 with reward r,
        get a transition s_t -> s_t+n with sum of intermediate rewards.

        Used in a variant of Deep Q-Learning
        """
        indexes = self.backend.sample_batch_transitions(batch_size, forward_steps=forward_steps)
        transition_tensors = self.backend.get_transitions_forward_steps(
            indexes, forward_steps=forward_steps, discount_factor=discount_factor
        )

        return Trajectories(
            num_steps=batch_size,
            num_envs=self.backend.num_envs,
            environment_information=None,
            transition_tensors={k: torch.from_numpy(v) for k, v in transition_tensors.items()},
            rollout_tensors={},
            extra_data={
                'forward_steps': forward_steps
            }
        ).to_transitions()

    def store_transition(self, frame, action, reward, done, extra_info=None):
        """ Store given transition in the backend """
        self.backend.store_transition(frame=frame, action=action, reward=reward, done=done, extra_info=extra_info)

    def update(self, rollout, batch_info):
        """ Don't need to update anything """
        pass


class CircularReplayBufferFactory(ReplayBufferFactory):
    """ Factory class for the CircularReplayBuffer """

    def __init__(self, buffer_capacity: int, buffer_initial_size: int,
                 frame_stack_compensation: bool = False, frame_history: int = 1):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack_compensation = frame_stack_compensation
        self.frame_history = frame_history

    def instantiate(self, environment):
        return CircularReplayBuffer(
            buffer_capacity=self.buffer_capacity,
            buffer_initial_size=self.buffer_initial_size,
            num_envs=environment.num_envs,
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            frame_stack_compensation=self.frame_stack_compensation,
            frame_history=self.frame_history
        )


def create(buffer_capacity: int, buffer_initial_size: int, frame_stack_compensation: bool = False,
           frame_history: int = 1):
    """ Vel factory function """
    return CircularReplayBufferFactory(
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        frame_stack_compensation=frame_stack_compensation,
        frame_history=frame_history
    )
