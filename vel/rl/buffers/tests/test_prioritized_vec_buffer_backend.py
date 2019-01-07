import collections
import gym
import gym.spaces
import nose.tools as t
import numpy as np
import numpy.testing as nt

from vel.exceptions import VelException
from vel.rl.buffers.backend.prioritized_vec_buffer_backend import PrioritizedCircularVecEnvBufferBackend


def get_halfempty_buffer_with_dones(frame_history=1):
    """ Return simple preinitialized buffer with some done's in there """
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)

    buffer = PrioritizedCircularVecEnvBufferBackend(
        buffer_capacity=20, num_envs=2, observation_space=observation_space, action_space=action_space,
        frame_history=frame_history
    )

    v1 = np.ones(8).reshape((2, 2, 2, 1))

    done_set = {2, 5, 10, 13, 18, 22, 28}

    for i in range(10):
        if i in done_set:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, True)
        else:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, False)

    return buffer


def get_filled_buffer_with_dones(frame_history=1):
    """ Return simple preinitialized buffer with some done's in there """
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)

    buffer = PrioritizedCircularVecEnvBufferBackend(
        buffer_capacity=20, num_envs=2, observation_space=observation_space, action_space=action_space,
        frame_history=frame_history
    )

    v1 = np.ones(8).reshape((2, 2, 2, 1))

    done_set = {2, 5, 10, 13, 18, 22, 28}

    for i in range(30):
        if i in done_set:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, True)
        else:
            buffer.store_transition(v1 * (i+1), 0, float(i)/2, False)

    return buffer


def get_large_filled_buffer_with_dones(frame_history=1):
    """ Return simple preinitialized buffer with some done's in there """
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)

    buffer = PrioritizedCircularVecEnvBufferBackend(
        buffer_capacity=2000, num_envs=2, observation_space=observation_space, action_space=action_space,
        frame_history=frame_history
    )

    v1 = np.ones(8).reshape((2, 2, 2, 1))

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
    buffer = get_filled_buffer_with_dones(frame_history=4)

    for i in range(100):
        probs, idxs, tree_idxs = buffer.sample_batch_transitions(6)
        buffer.get_transitions(idxs)

        nt.assert_array_equal(probs, np.ones((6, 2)))

    with t.assert_raises(VelException):
        buffer.get_transitions(np.array([[10, 10]]))


def test_sampling_is_correct_small_buffer():
    """ Check if sampling multiple times we don't get incorrect values"""
    buffer = get_halfempty_buffer_with_dones(frame_history=4)

    for i in range(100):
        probs, idxs, tree_idxs = buffer.sample_batch_transitions(6)
        buffer.get_transitions(idxs)

        nt.assert_array_equal(np.array(probs), np.ones((6, 2)))
        assert np.all(idxs <= 10)

    with t.assert_raises(VelException):
        buffer.get_transitions(np.array([[10, 10]]))


def test_prioritized_sampling_probabilities():
    """ Check if sampling probabilities are more or less correct in the sampling results """
    buffer = get_large_filled_buffer_with_dones(frame_history=4)

    zero_tree_idx = [tree.tree_index_for_index(0) for tree in buffer.segment_trees]

    # Give much more priority to specified element
    high_prio = [100.0 for _ in zero_tree_idx]
    buffer.update_priority(zero_tree_idx, high_prio)

    counter = collections.Counter()

    for i in range(1000):
        probs, idxs, tree_idxs = buffer.sample_batch_transitions(6)

        for idx in idxs:
            counter.update(idx)

    # Element with the highest priority is the one that happens the most often
    t.eq_(counter[0], max(counter.values()))

    # Make priority low now
    buffer.update_priority(zero_tree_idx, [0.01 for _ in zero_tree_idx])

    counter = collections.Counter()

    for i in range(1000):
        probs, idxs, tree_idxs = buffer.sample_batch_transitions(6)

        for idx in idxs:
            counter.update(idx)

    # At least half of the element have greater counts than zero
    t.assert_greater(np.mean([1 if counter.get(i, 0) > counter.get(0, 0) else 0 for i in range(2000)]), 0.5)
