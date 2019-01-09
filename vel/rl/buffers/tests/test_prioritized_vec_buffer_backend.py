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


def get_filled_buffer_frame_stack(frame_stack=4, frame_dim=1):
    """ Return a preinitialized buffer with frame stack implemented """
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, frame_dim * frame_stack), dtype=int)
    action_space = gym.spaces.Discrete(4)

    buffer = PrioritizedCircularVecEnvBufferBackend(
        buffer_capacity=20, num_envs=2, observation_space=observation_space, action_space=action_space,
        frame_stack_compensation=True, frame_history=frame_stack
    )

    v1 = np.ones(8 * frame_dim).reshape((2, 2, 2, frame_dim))
    done_set = {2, 5, 10, 13, 18, 22, 28}

    # simple buffer of previous frames to simulate frame stack
    item_array = []

    for i in range(30):
        item = v1.copy()

        item[:, 0] *= (i+1)
        item[:, 1] *= 10 * (i+1)

        done_array = np.array([i in done_set, (i+1) in done_set], dtype=bool)

        item_array.append(item)

        if len(item_array) < frame_stack:
            item_concatenated = np.concatenate([item] * frame_stack, axis=-1)
        else:
            item_concatenated = np.concatenate(item_array[-frame_stack:], axis=-1)

        buffer.store_transition(item_concatenated, 0, float(i) / 2, done_array)

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


def test_frame_stack_compensation_single_dim():
    buffer = get_filled_buffer_frame_stack(frame_stack=4, frame_dim=1)

    observations_1 = buffer.get_transition(frame_idx=0, env_idx=0)['observations']
    observations_2 = buffer.get_transition(frame_idx=1, env_idx=0)['observations']
    observations_3 = buffer.get_transition(frame_idx=2, env_idx=0)['observations']

    nt.assert_array_almost_equal(
        observations_1, np.array([[[0,   0,  20,  21],
                                   [0,   0,  20,  21]],
                                  [[0,   0, 200, 210],
                                   [0,   0, 200, 210]]])
    )

    nt.assert_array_almost_equal(
        observations_2, np.array([[[0,  20,  21,  22],
                                   [0,  20,  21,  22]],
                                  [[0, 200, 210, 220],
                                   [0, 200, 210, 220]]])
    )

    nt.assert_array_almost_equal(
        observations_3, np.array([[[ 20,  21,  22,  23],
                                   [ 20,  21,  22,  23]],
                                  [[200, 210, 220, 230],
                                   [200, 210, 220, 230]]])
    )


def test_frame_stack_compensation_multi_dim():
    buffer = get_filled_buffer_frame_stack(frame_stack=4, frame_dim=2)

    observations_1 = buffer.get_transition(frame_idx=0, env_idx=0)['observations']
    observations_2 = buffer.get_transition(frame_idx=1, env_idx=0)['observations']
    observations_3 = buffer.get_transition(frame_idx=2, env_idx=0)['observations']

    nt.assert_array_almost_equal(
        observations_1, np.array([[[0,   0,   0,   0,  20,  20,  21,  21],
                                   [0,   0,   0,   0,  20,  20,  21,  21]],
                                  [[0,   0,   0,   0, 200, 200, 210, 210],
                                   [0,   0,   0,   0, 200, 200, 210, 210]]])
    )

    nt.assert_array_almost_equal(
        observations_2, np.array([[[0, 0, 20, 20, 21, 21, 22, 22],
                                   [0, 0, 20, 20, 21, 21, 22, 22]],
                                  [[0, 0, 200, 200, 210, 210, 220, 220],
                                   [0, 0, 200, 200, 210, 210, 220, 220]]])
    )

    nt.assert_array_almost_equal(
        observations_3, np.array([[[20, 20, 21, 21, 22, 22, 23, 23],
                                   [20, 20, 21, 21, 22, 22, 23, 23]],
                                  [[200, 200, 210, 210, 220, 220, 230, 230],
                                   [200, 200, 210, 210, 220, 220, 230, 230]]])
    )
