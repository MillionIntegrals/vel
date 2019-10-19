import gym
import numpy as np
import random

from .circular_buffer_backend import CircularBufferBackend
from .segment_tree import SegmentTree


class PrioritizedCircularBufferBackend:
    """ Backend behind the prioritized replay buffer using circular buffer """
    def __init__(self, buffer_capacity: int, observation_space: gym.Space, action_space: gym.Space, extra_data=None):
        self.deque = CircularBufferBackend(buffer_capacity, observation_space, action_space, extra_data=extra_data)
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
        return self.deque.get_transitions(indexes, history)

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
