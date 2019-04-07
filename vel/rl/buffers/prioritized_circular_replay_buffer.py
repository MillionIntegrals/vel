import gym
import numpy as np
import torch
import typing

from vel.api import Schedule
from vel.rl.api import ReplayBuffer, Trajectories, Transitions
from .backend.prioritized_vec_buffer_backend import PrioritizedCircularVecEnvBufferBackend


class PrioritizedCircularReplayBuffer(ReplayBuffer):
    """
    Replay buffer that supports prioritized experience replay as an overlay over a circular buffer.

    Frame stack compensation - if environment has a framestack built in, we will store only the last frame
    """

    def __init__(self, buffer_capacity: int, buffer_initial_size: int, num_envs: int, observation_space: gym.Space,
                 action_space: gym.Space, priority_exponent: float, priority_weight: Schedule, priority_epsilon: float,
                 frame_stack_compensation: bool = False, frame_history: int = 1):
        super().__init__()

        self.buffer_initial_size = buffer_initial_size

        self.priority_exponent = priority_exponent
        self.priority_weight = priority_weight
        self.priority_epsilon = priority_epsilon

        self.backend = PrioritizedCircularVecEnvBufferBackend(
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

    def _get_transitions(self, probs, indexes, tree_idxs, batch_info, forward_steps=1, discount_factor=1.0):
        """ Return batch of frames for given indexes """
        if forward_steps > 1:
            transition_arrays = self.backend.get_transitions_forward_steps(indexes, forward_steps, discount_factor)
        else:
            transition_arrays = self.backend.get_transitions(indexes)

        priority_weight = self.priority_weight.value(batch_info['progress'])

        # Normalize by sum of all probs
        probs = probs / np.array([s.total() for s in self.backend.segment_trees], dtype=float).reshape(1, -1)
        capacity = self.backend.current_size
        weights = (capacity * probs) ** (-priority_weight)
        weights = weights / weights.max(axis=0, keepdims=True)

        transition_arrays['weights'] = weights
        transition_tensors = {k: torch.from_numpy(v) for k, v in transition_arrays.items()}

        transitions = Trajectories(
            num_steps=indexes.shape[0],
            num_envs=indexes.shape[1],
            environment_information=None,
            transition_tensors=transition_tensors,
            rollout_tensors={},
            extra_data={
                'tree_idxs': tree_idxs
            }
        )

        return transitions.to_transitions()

    def sample_transitions(self, batch_size, batch_info) -> Transitions:
        """ Sample batch of transitions and return them """
        probs, indexes, tree_idxs = self.backend.sample_batch_transitions(batch_size)

        return self._get_transitions(probs, indexes, tree_idxs, batch_info)

    def sample_forward_transitions(self, batch_size, batch_info,
                                   forward_steps: int, discount_factor: float) -> Transitions:
        """
        Sample transitions from replay buffer with _forward steps_.
        That is, instead of getting a transition s_t -> s_t+1 with reward r,
        get a transition s_t -> s_t+n with sum of intermediate rewards.

        Used in a variant of Deep Q-Learning
        """
        probs, indexes, tree_idxs = self.backend.sample_batch_transitions(batch_size, forward_steps)

        return self._get_transitions(
            probs, indexes, tree_idxs, batch_info
        )

    def sample_trajectories(self, rollout_length, batch_info):
        """ Sample batch of trajectories and return them """
        raise NotImplementedError("There is no good idea so far how to sample trajectories in a prioritized way")

    def store_transition(self, frame, action, reward, done, extra_info=None):
        """ Store given transition in the backend """
        self.backend.store_transition(frame=frame, action=action, reward=reward, done=done, extra_info=extra_info)

    def update(self, rollout, batch_info):
        tree_idxs = rollout.extra_data['tree_idxs']
        errors = batch_info['errors'].reshape(tree_idxs.shape)

        weights = (errors + self.priority_epsilon) ** self.priority_exponent

        for idx, priority in zip(tree_idxs, weights):
            self.backend.update_priority(idx, priority)


class PrioritizedCircularVecEnvBufferBackendFactory:
    """ Factory class for the CircularVecEnvBufferBackend """

    def __init__(self, buffer_capacity: int, buffer_initial_size: int, priority_exponent: float,
                 priority_weight: Schedule, priority_epsilon: float, frame_stack_compensation: bool = False,
                 frame_history: int = 1):
        self.buffer_capacity = buffer_capacity
        self.buffer_initial_size = buffer_initial_size
        self.frame_stack_compensation = frame_stack_compensation
        self.frame_history = frame_history
        self.priority_exponent = priority_exponent
        self.priority_weight = priority_weight
        self.priority_epsilon = priority_epsilon

    def instantiate(self, environment):
        return PrioritizedCircularReplayBuffer(
            buffer_capacity=self.buffer_capacity,
            buffer_initial_size=self.buffer_initial_size,
            num_envs=environment.num_envs,
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            priority_exponent=self.priority_exponent,
            priority_weight=self.priority_weight,
            priority_epsilon=self.priority_epsilon,
            frame_stack_compensation=self.frame_stack_compensation,
            frame_history=self.frame_history
        )


def create(buffer_capacity: int, buffer_initial_size: int, priority_exponent: float, priority_weight: Schedule,
           priority_epsilon: float, frame_stack_compensation: bool = False, frame_history: int = 1):
    """ Vel factory function """
    return PrioritizedCircularVecEnvBufferBackendFactory(
        buffer_capacity=buffer_capacity,
        buffer_initial_size=buffer_initial_size,
        priority_exponent=priority_exponent,
        priority_weight=priority_weight,
        priority_epsilon=priority_epsilon,
        frame_stack_compensation=frame_stack_compensation,
        frame_history=frame_history
    )
