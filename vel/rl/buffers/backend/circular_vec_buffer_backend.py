import gym
import numpy as np

from vel.exceptions import VelException


def take_along_axis(large_array, indexes):
    """ Take along axis """
    # Reshape indexes into the right shape
    if len(large_array.shape) > len(indexes.shape):
        indexes = indexes.reshape(indexes.shape + tuple([1] * (len(large_array.shape) - len(indexes.shape))))

    return np.take_along_axis(large_array, indexes, axis=0)


class CircularVecEnvBufferBackend:
    """
    Backend class for replay buffer that uses a circular buffer - new experience overwrites the oldest one
    Version supporting multiple environments.

    Frame stack compensation - if environment has a framestack built in, we will store only the last action
    """

    def __init__(self, buffer_capacity: int, num_envs: int, observation_space: gym.Space, action_space: gym.Space,
                 frame_stack_compensation: bool=False, frame_history: int=1):
        # Maximum number of items in the buffer
        self.buffer_capacity = buffer_capacity

        self.frame_stack_compensation = frame_stack_compensation

        # Number of parallel envs to record
        self.num_envs = num_envs

        # How many elements have been inserted in the buffer
        self.current_size = 0

        # Index of last inserted element
        self.current_idx = -1

        # How many frames are stacked into each observation
        self.frame_history = frame_history

        # Data buffers
        if self.frame_stack_compensation:
            self.state_buffer = np.zeros(
                [self.buffer_capacity, self.num_envs] + list(observation_space.shape)[:-1] +
                [observation_space.shape[-1] // self.frame_history],
                dtype=observation_space.dtype
            )
        else:
            self.state_buffer = np.zeros(
                [self.buffer_capacity, self.num_envs] + list(observation_space.shape),
                dtype=observation_space.dtype
            )

        self.action_buffer = np.zeros(
            [self.buffer_capacity, self.num_envs] + list(action_space.shape), dtype=action_space.dtype
        )
        self.reward_buffer = np.zeros([self.buffer_capacity, self.num_envs], dtype=np.float32)
        self.dones_buffer = np.zeros([self.buffer_capacity, self.num_envs], dtype=bool)

        self.extra_data = {}

        # Just a sentinel to simplify further calculations
        self.dones_buffer[self.current_idx] = True

    def store_transition(self, frame, action, reward, done, extra_info=None):
        """ Store given transition in the backend """
        self.current_idx = (self.current_idx + 1) % self.buffer_capacity

        if self.frame_stack_compensation:
            # Compensate for frame stack built into the environment
            idx_range = np.arange(-frame.shape[-1] // self.frame_history, 0)
            frame = np.take(frame, indices=idx_range, axis=-1)

        self.state_buffer[self.current_idx] = frame

        self.action_buffer[self.current_idx] = action
        self.reward_buffer[self.current_idx] = reward
        self.dones_buffer[self.current_idx] = done

        if extra_info is not None:
            for name in extra_info:
                if name not in self.extra_data:
                    assert self.current_size == 0, f"New data {name} encountered in the middle of the training"
                    array = extra_info[name]

                    self.extra_data[name] = np.zeros([self.buffer_capacity] + list(array.shape), dtype=array.dtype)

                self.extra_data[name][self.current_idx] = extra_info[name]

        if self.current_size < self.buffer_capacity:
            self.current_size += 1

        return self.current_idx

    def get_frame_with_future(self, frame_idx, env_idx):
        """ Return frame from the buffer together with the next frame """
        if frame_idx == self.current_idx:
            raise VelException("Cannot provide enough future for the frame")

        past_frame = self.get_frame(frame_idx, env_idx)

        # if self.frame_history > 1:
        #     assert self.state_buffer.shape[-1] == 1, \
        #         "State buffer must have last dimension of 1 if we want frame history"

        if not self.dones_buffer[frame_idx, env_idx]:
            next_idx = (frame_idx + 1) % self.buffer_capacity
            next_frame = self.state_buffer[next_idx, env_idx]
        else:
            next_idx = (frame_idx + 1) % self.buffer_capacity
            next_frame = np.zeros_like(self.state_buffer[next_idx, env_idx])

        if self.frame_history > 1:
            future_frame = np.concatenate([
                past_frame.take(indices=np.arange(1, past_frame.shape[-1]), axis=-1), next_frame
            ], axis=-1)
        else:
            future_frame = next_frame

        return past_frame, future_frame

    def get_frame(self, frame_idx, env_idx):
        """ Return frame from the buffer """
        if frame_idx >= self.current_size:
            raise VelException("Requested frame beyond the size of the buffer")

        # if self.frame_history > 1:
        #     assert self.state_buffer.shape[-1] == 1, \
        #         "State buffer must have last dimension of 1 if we want frame history"

        accumulator = []

        last_frame = self.state_buffer[frame_idx, env_idx]

        accumulator.append(last_frame)

        for i in range(self.frame_history - 1):
            prev_idx = (frame_idx - 1) % self.buffer_capacity

            if prev_idx == self.current_idx:
                raise VelException("Cannot provide enough history for the frame")
            elif self.dones_buffer[prev_idx, env_idx]:
                # If previous frame was done - just append zeroes
                accumulator.append(np.zeros_like(last_frame))
            else:
                frame_idx = prev_idx
                accumulator.append(self.state_buffer[frame_idx, env_idx])

        # We're pushing the elements in reverse order
        return np.concatenate(accumulator[::-1], axis=-1)

    def get_transition(self, frame_idx, env_idx):
        """ Single transition with given index """
        past_frame, future_frame = self.get_frame_with_future(frame_idx, env_idx)

        data_dict = {
            'observations': past_frame,
            'observations_next': future_frame,
            'actions': self.action_buffer[frame_idx, env_idx],
            'rewards': self.reward_buffer[frame_idx, env_idx],
            'dones': self.dones_buffer[frame_idx, env_idx],
        }

        for name in self.extra_data:
            data_dict[name] = self.extra_data[name][frame_idx, env_idx]

        return data_dict

    def get_transitions(self, indexes):
        """ Get dictionary of transition data """
        assert indexes.shape[1] == self.state_buffer.shape[1], \
            "Must have the same number of indexes as there are environments"

        frame_batch_shape = (
                [indexes.shape[0], indexes.shape[1]]
                + list(self.state_buffer.shape[2:-1])
                + [self.state_buffer.shape[-1] * self.frame_history]
        )

        past_frame_buffer = np.zeros(frame_batch_shape, dtype=self.state_buffer.dtype)
        future_frame_buffer = np.zeros(frame_batch_shape, dtype=self.state_buffer.dtype)

        for buffer_idx, frame_row in enumerate(indexes):
            for env_idx, frame_idx in enumerate(frame_row):
                past_frame_buffer[buffer_idx, env_idx], future_frame_buffer[buffer_idx, env_idx] = (
                    self.get_frame_with_future(frame_idx, env_idx)
                )

        actions = take_along_axis(self.action_buffer, indexes)
        rewards = take_along_axis(self.reward_buffer, indexes)
        dones = take_along_axis(self.dones_buffer, indexes)

        transition_tensors = {
            'observations': past_frame_buffer,
            'actions': actions,
            'rewards': rewards,
            'observations_next': future_frame_buffer,
            'dones': dones.astype(np.float32),
        }

        for name in self.extra_data:
            transition_tensors[name] = take_along_axis(self.extra_data[name], indexes)

        return transition_tensors

    def get_trajectories(self, indexes, rollout_length):
        """ Return batch consisting of *consecutive* transitions """
        # assert indexes.shape[0] > 1, "There must be multiple indexes supplied"
        assert rollout_length > 1, "Rollout length must be greater than 1"

        batch_indexes = (
                indexes.reshape(1, indexes.shape[0]) - np.arange(rollout_length - 1, -1, -1).reshape(rollout_length, 1)
        )

        return self.get_transitions(batch_indexes)

    def sample_batch_transitions(self, batch_size):
        """ Return indexes of next sample"""
        results = []

        for i in range(self.num_envs):
            results.append(self.sample_uniform_single_env(batch_size))

        return np.stack(results, axis=-1)

    def sample_batch_trajectories(self, rollout_length):
        """ Return indexes of next random rollout """
        results = []

        for i in range(self.num_envs):
            results.append(self.sample_rollout_single_env(rollout_length))

        return np.stack(results, axis=-1)

    def sample_rollout_single_env(self, rollout_length):
        """ Return indexes of next sample"""
        # Sample from up to total size
        if self.current_size < self.buffer_capacity:
            if rollout_length + 1 > self.current_size:
                raise VelException("Not enough elements in the buffer to sample the rollout")

            # -1 because we cannot take the last one
            return np.random.choice(self.current_size - rollout_length) + rollout_length - 1
        else:
            if rollout_length + self.frame_history > self.current_size:
                raise VelException("Not enough elements in the buffer to sample the rollout")

            candidate = np.random.choice(self.buffer_capacity)

            # These are the elements we cannot draw, as then we don't have enough history
            forbidden_ones = (
                    np.arange(self.current_idx, self.current_idx + self.frame_history + rollout_length - 1)
                    % self.buffer_capacity
            )

            # Exclude these frames for learning as they may have some part of history overwritten
            while candidate in forbidden_ones:
                candidate = np.random.choice(self.buffer_capacity)

            return candidate

    def sample_uniform_single_env(self, batch_size):
        """ Return indexes of next sample"""
        # Sample from up to total size
        if self.current_size < self.buffer_capacity:
            # -1 because we cannot take the last one
            return np.random.choice(self.current_size - 1, batch_size, replace=False)
        else:
            candidate = np.random.choice(self.buffer_capacity, batch_size, replace=False)

            forbidden_ones = (
                    np.arange(self.current_idx, self.current_idx + self.frame_history) % self.buffer_capacity
            )

            # Exclude these frames for learning as they may have some part of history overwritten
            while any(x in candidate for x in forbidden_ones):
                candidate = np.random.choice(self.buffer_capacity, batch_size, replace=False)

            return candidate
