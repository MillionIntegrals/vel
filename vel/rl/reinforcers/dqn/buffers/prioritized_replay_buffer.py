import gym
import numpy as np
import random
import torch


from vel.api.base import Schedule, Model
from ..dqn_reinforcer import DqnBufferBase
from .deque_buffer import DequeBufferBackend


# Segment tree implementation taken from https://github.com/Kaixhin/Rainbow/blob/master/memory.py
class SegmentTree:
    """
    Segment tree data structure where parent node values are sum/max of children node values
    """
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
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

    # def append(self, data, value):
    def append(self, value):
        # self.data[self.index] = data  # Store data in underlying data structure
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

    def tree_index_for_index(self, index):
        return index + self.size - 1

    def total(self):
        return self.sum_tree[0]


class PrioritizedReplayBackend:
    """ Backend behind the prioritized replay buffer """
    def __init__(self, buffer_capacity: int, frame_shape: tuple):
        self.deque = DequeBufferBackend(buffer_capacity, frame_shape)
        self.segment_tree = SegmentTree(buffer_capacity)

    def store_transition(self, frame, action, reward, done):
        """ Store given transition in the backend """
        self.deque.store_transition(frame, action, reward, done)

        # We add element with max priority to the tree
        self.segment_tree.append(self.segment_tree.max)

    def get_frame(self, idx, history):
        """ Return frame from the buffer """
        return self.deque.get_frame(idx, history)

    def get_frame_with_future(self, idx, history):
        """ Return frame from the buffer together with the next frame """
        return self.deque.get_frame_with_future(idx, history)

    def get_batch(self, indexes, history):
        """ Return batch of frames for given indexes """
        return self.deque.get_batch(indexes, history)

    def update_priority(self, tree_idx, priority):
        """ Update priorities of the elements in the tree """
        self.segment_tree.update(tree_idx, priority)

    def sample_batch_prioritized(self, batch_size, history):
        """ Return indexes of the next sample in from prioritized distribution """
        p_total = self.segment_tree.total()
        segment = p_total / batch_size

        # Get batch of valid samples
        batch = [self._get_sample_from_segment(segment, i, history) for i in range(batch_size)]
        probs, idxs, tree_idxs = zip(*batch)
        return probs, np.array(idxs), tree_idxs

    def _get_sample_from_segment(self, segment, i, history):
        valid = False

        prob = None
        idx = None
        tree_idx = None

        while not valid:
            # Uniformly sample an element from within a segment
            sample = random.uniform(i * segment, (i + 1) * segment)

            # Retrieve sample from tree with un-normalised probability
            prob, idx, tree_idx = self.segment_tree.find(sample)

            # Resample if transition straddled current index or probablity 0
            if (self.segment_tree.index - idx) % self.segment_tree.size > 1 and \
                    (idx - self.segment_tree.index) % self.segment_tree.size >= history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        return prob, idx, tree_idx

    @property
    def current_size(self):
        """ Return current size of the replay buffer """
        return self.deque.current_size

    @property
    def current_idx(self):
        """ Return current index """
        return self.deque.current_idx


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
            frame_shape=environment.observation_space.shape
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
        probs, indexes, tree_idxs = self.backend.sample_batch_prioritized(batch_size, self.frame_stack)
        observations, actions, rewards, observations_tplus1, dones = self.backend.get_batch(indexes, self.frame_stack)

        # Normalize weights properly
        priority_weight = self.priority_weight_schedule.value(batch_info['progress'])

        probs = np.stack(probs) / self.backend.segment_tree.total()
        capacity = self.backend.deque.current_size
        weights = (capacity * probs) ** (-priority_weight)
        weights = weights / weights.max()

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'observations_tplus1': observations_tplus1,
            'weights': weights,
            'tree_idxs': tree_idxs
        }

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

