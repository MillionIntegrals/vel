# Segment tree implementation taken from https://github.com/Kaixhin/Rainbow/blob/master/memory.py
import gym
import numpy as np
import random
import torch

from collections import namedtuple

from vel.api.base import Schedule, Model
from ..dqn_reinforcer import DqnBufferBase

# Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
Transition = namedtuple('Transition', ('observation', 'action', 'reward', 'done'))


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree:
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
        self.data = [None] * size  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return self.sum_tree[index], data_index, index  # Return value, data index, tree index

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


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

        self.current_index = -1
        self.total_size = 0
        self.t = 0  # Internal episode timestep counter
        self.transitions = SegmentTree(self.buffer_capacity)

        self.device = None
        self.last_observation = None
        self.frame_shape = None
        self.blank_transition = None

    def initialize(self, environment: gym.Env, device: torch.device):
        """ Initialze buffer for operation """
        self.device = device
        self.last_observation = environment.reset()
        self.frame_shape = list(environment.observation_space.shape)
        self.blank_transition = Transition(np.zeros_like(self.last_observation), 0, reward=0.0, done=False)

    def rollout(self, environment: gym.Env, model: Model, epsilon_value: float) -> dict:
        """ Evaluate model and proceed one step forward with the environment. Store result in the replay buffer """
        last_observation_frame = self._get_current_frame()
        observation_tensor = torch.from_numpy(last_observation_frame[None]).to(self.device)
        action = model.step(observation_tensor, epsilon_value).item()

        observation, reward, done, info = environment.step(action)
        observation = observation[:]

        transition = Transition(self.last_observation, action, reward, done)

        self.transitions.append(transition, self.transitions.max)

        if self.total_size < self.buffer_capacity:
            self.total_size += 1

        self.current_index = (self.current_index + 1) % self.buffer_capacity

        # Usual, reset on done
        if done:
            observation = environment.reset()[:]

        self.last_observation = observation

        return info.get('episode')

    def sample(self, batch_info, batch_size) -> dict:
        """ Calculate random sample from the replay buffer """
        priority_weight = self.priority_weight_schedule.value(batch_info['progress'])

        p_total = self.transitions.total()
        segment = p_total / batch_size
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, observations, actions, rewards, observations_tplus1, dones = zip(*batch)

        probs = np.stack(probs) / p_total
        capacity = self.buffer_capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** (-priority_weight)
        weights = weights / weights.max()

        return {
            'weights': weights,
            'tree_idxs': tree_idxs,
            'observations': np.stack(observations),
            'observations_tplus1': np.stack(observations_tplus1),
            'actions': np.stack(actions),
            'rewards': np.stack(rewards),
            'dones': np.stack(dones)
        }

    def update(self, sample, errors):
        errors.pow_(self.priority_exponent)

        for idx, priority in zip(sample['tree_idxs'], errors):
            self.transitions.update(idx, priority.item())

    def _get_sample_from_segment(self, segment, i):
        valid = False

        while not valid:
            sample = random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability

            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.buffer_capacity > 1 and (
                    idx - self.transitions.index) % self.buffer_capacity >= self.frame_stack and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idx)

        observation = np.concatenate([x.observation for x in transitions[:self.frame_stack]], axis=-1)
        observation_tplus1 = np.concatenate([x.observation for x in transitions[1:self.frame_stack+1]], axis=-1)

        action = transitions[self.frame_stack - 1].action
        reward = transitions[self.frame_stack - 1].reward
        done = transitions[self.frame_stack - 1].done

        return prob, idx, tree_idx, observation, action, reward, observation_tplus1, done

        # Returns a transition with blank states where appropriate

    def _get_transitions(self, idx):
        transitions = [None] * (self.frame_stack + 1)

        transitions[self.frame_stack - 1] = self.transitions.get(idx)
        transitions[self.frame_stack] = self.transitions.get(idx+1)

        for t in range(self.frame_stack - 2, -1, -1):  # e.g. 2 1 0
            candidate_transition = self.transitions.get(idx - self.frame_stack + 1 + t)
            if candidate_transition is None or candidate_transition.done:
                break

            transitions[t] = candidate_transition

        for i in range(len(transitions)):
            if transitions[i] is None:
                transitions[i] = self.blank_transition

        return transitions

    def is_ready(self) -> bool:
        """ If buffer is ready for training """
        return self.total_size >= self.buffer_initial_size

    def _get_current_frame(self):
        """ Return the last frame for evaluating the environment """
        accumulator = []
        accumulator.append(self.last_observation)

        idx = self.current_index
        transition = self.transitions.get(idx)

        for i in range(self.frame_stack - 1):
            if transition is None:
                accumulator.append(np.zeros_like(self.last_observation))
            elif transition.done:
                accumulator.append(np.zeros_like(self.last_observation))
            else:
                idx = idx - 1
                accumulator.append(transition.observation)
                transition = self.transitions.get(idx)

        # We're pushing the elements in reverse order
        return np.concatenate(accumulator[::-1], axis=-1)


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

