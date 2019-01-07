import collections
import gym
import nose.tools as t
import numpy as np
import numpy.testing as nt

from vel.exceptions import VelException
from vel.rl.buffers.backend.prioritized_buffer_backend import PrioritizedCircularBufferBackend


def get_halfempty_buffer_with_dones():
    """ Return simple preinitialized buffer with some done's in there """
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)
    buffer = PrioritizedCircularBufferBackend(20, observation_space, action_space)

    v1 = np.ones(4).reshape((2, 2, 1))

    done_set = {2, 5, 10, 13, 18, 22, 28}

    for i in range(10):
        if i in done_set:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, True)
        else:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, False)

    return buffer


def get_filled_buffer_with_dones():
    """ Return simple preinitialized buffer with some done's in there """
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)
    buffer = PrioritizedCircularBufferBackend(20, observation_space, action_space)

    v1 = np.ones(4).reshape((2, 2, 1))

    done_set = {2, 5, 10, 13, 18, 22, 28}

    for i in range(30):
        if i in done_set:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, True)
        else:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, False)

    return buffer


def get_large_filled_buffer_with_dones():
    """ Return simple preinitialized buffer with some done's in there """
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)
    buffer = PrioritizedCircularBufferBackend(2000, observation_space, action_space)

    v1 = np.ones(4).reshape((2, 2, 1))

    done_increment = 2
    done_index = 2

    for i in range(3000):
        if i == done_index:
            done_increment += 1
            done_index += done_increment

            buffer.store_transition(v1 * (i+1), 0, float(i)/2, True)
        else:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, False)

    return buffer


def test_sampling_is_correct():
    """ Check if sampling multiple times we don't get incorrect values"""
    buffer = get_filled_buffer_with_dones()

    for i in range(100):
        probs, idxs, tree_idxs = buffer.sample_batch_prioritized(6, history=4)
        buffer.get_batch(idxs, history=4)

        nt.assert_array_equal(np.array(probs), np.ones(6))

    with t.assert_raises(VelException):
        buffer.get_batch(np.array([10]), history=4)


def test_sampling_is_correct_small_buffer():
    """ Check if sampling multiple times we don't get incorrect values"""
    buffer = get_halfempty_buffer_with_dones()

    for i in range(100):
        probs, idxs, tree_idxs = buffer.sample_batch_prioritized(6, history=4)
        buffer.get_batch(idxs, history=4)

        nt.assert_array_equal(np.array(probs), np.ones(6))
        assert np.all(idxs <= 10)

    with t.assert_raises(VelException):
        buffer.get_batch(np.array([10]), history=4)


def test_prioritized_sampling_probabilities():
    """ Check if sampling probabilities are more or less correct in the sampling results """
    buffer = get_large_filled_buffer_with_dones()

    zero_tree_idx = buffer.segment_tree.tree_index_for_index(0)

    # Give much more priority to specified element
    high_prio = 100.0
    buffer.update_priority(zero_tree_idx, high_prio)

    counter = collections.Counter()

    for i in range(1000):
        probs, idxs, tree_idxs = buffer.sample_batch_prioritized(6, history=4)
        counter.update(idxs)

    # Element with the highest priority is the one that happens the most often
    t.eq_(counter[0], max(counter.values()))

    # Go back to original priority
    buffer.update_priority(zero_tree_idx, 0.1)

    counter = collections.Counter()

    for i in range(1000):
        probs, idxs, tree_idxs = buffer.sample_batch_prioritized(6, history=4)
        counter.update(idxs)

    # At least half of the element have greater counts than zero
    t.assert_greater(np.mean([1 if counter.get(i, 0) > counter.get(0, 0) else 0 for i in range(2000)]), 0.7)
