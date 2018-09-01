import gym
import numpy as np
import random

from .deque_backend import DequeBufferBackend


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
    def __init__(self, buffer_capacity: int, observation_space: gym.Space, action_space: gym.Space, extra_data=None):
        self.deque = DequeBufferBackend(buffer_capacity, observation_space, action_space, extra_data=extra_data)
        self.segment_tree = SegmentTree(buffer_capacity)

    def store_transition(self, frame, action, reward, done, extra_info=None):
        """ Store given transition in the backend """
        self.deque.store_transition(frame, action, reward, done, extra_info=extra_info)

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
