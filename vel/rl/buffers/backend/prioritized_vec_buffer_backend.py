import gym
import random
import numpy as np

from .circular_vec_buffer_backend import CircularVecEnvBufferBackend
from .segment_tree import SegmentTree


class PrioritizedCircularVecEnvBufferBackend:
    """
    Prioritized replay buffer version of CircularVecEnvBufferBackend
    """

    def __init__(self, buffer_capacity: int, num_envs: int, observation_space: gym.Space, action_space: gym.Space,
                 frame_stack_compensation: bool = False, frame_history: int = 1):
        self.deque = CircularVecEnvBufferBackend(
            buffer_capacity=buffer_capacity,
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
            frame_stack_compensation=frame_stack_compensation,
            frame_history=frame_history
        )

        self.segment_trees = [SegmentTree(buffer_capacity) for _ in range(num_envs)]

    def store_transition(self, frame, action, reward, done, extra_info=None):
        """ Store given transition in the backend """
        self.deque.store_transition(frame, action, reward, done, extra_info=extra_info)

        # We add element with max priority to all the trees
        for segment_tree in self.segment_trees:
            segment_tree.append(segment_tree.max)

    def get_frame_with_future(self, frame_idx, env_idx):
        """ Return frame from the buffer together with the next frame """
        return self.deque.get_frame_with_future(frame_idx, env_idx)

    def get_frame(self, frame_idx, env_idx):
        """ Return frame from the buffer """
        return self.deque.get_frame(frame_idx, env_idx)

    def get_transition(self, frame_idx, env_idx):
        """ Single transition with given index """
        return self.deque.get_transition(frame_idx, env_idx)

    def get_trajectories(self, indexes, rollout_length):
        """ Return batch consisting of *consecutive* transitions """
        return self.deque.get_trajectories(indexes, rollout_length)

    def get_transitions(self, indexes):
        """ Get dictionary of transition data """
        return self.deque.get_transitions(indexes)

    def get_transitions_forward_steps(self, indexes, forward_steps, discount_factor):
        """ Get dictionary of transition data """
        return self.deque.get_transitions_forward_steps(indexes, forward_steps, discount_factor)

    def sample_batch_transitions(self, batch_size, forward_steps=1):
        """ Return indexes of next sample"""
        batches = [
            self._sample_batch_prioritized(
                segment_tree, batch_size, self.deque.frame_history, forward_steps=forward_steps
            )
            for segment_tree in self.segment_trees
        ]

        probs, idxs, tree_idxs = zip(*batches)

        return np.stack(probs, axis=1).astype(float), np.stack(idxs, axis=1), np.stack(tree_idxs, axis=1)

    def update_priority(self, tree_idx_list, priority_list):
        """ Update priorities of the elements in the tree """
        for tree_idx, priority, segment_tree in zip(tree_idx_list, priority_list, self.segment_trees):
            segment_tree.update(tree_idx, priority)

    @property
    def current_size(self):
        """ Return current size of the replay buffer """
        return self.deque.current_size

    @property
    def current_idx(self):
        """ Return current index """
        return self.deque.current_idx

    def _sample_batch_prioritized(self, segment_tree, batch_size, history, forward_steps=1):
        """ Return indexes of the next sample in from prioritized distribution """
        p_total = segment_tree.total()
        segment = p_total / batch_size

        # Get batch of valid samples
        batch = [
            self._get_sample_from_segment(segment_tree, segment, i, history, forward_steps)
            for i in range(batch_size)
        ]

        probs, idxs, tree_idxs = zip(*batch)
        return np.array(probs), np.array(idxs), np.array(tree_idxs)

    def _get_sample_from_segment(self, segment_tree, segment, i, history, forward_steps):
        valid = False

        prob = None
        idx = None
        tree_idx = None

        while not valid:
            # Uniformly sample an element from within a segment
            sample = random.uniform(i * segment, (i + 1) * segment)

            # Retrieve sample from tree with un-normalised probability
            prob, idx, tree_idx = segment_tree.find(sample)

            # Resample if transition straddled current index or probablity 0
            if (segment_tree.index - idx) % segment_tree.size > forward_steps and \
                    (idx - segment_tree.index) % segment_tree.size >= history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        return prob, idx, tree_idx
