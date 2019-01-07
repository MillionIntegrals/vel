import gym
import numpy as np

from vel.exceptions import VelException


class CircularBufferBackend:
    """ Backend class for replay buffer that uses a circular buffer - new experience overwrites the oldest one """

    def __init__(self, buffer_capacity: int, observation_space: gym.Space, action_space: gym.Space, extra_data=None):
        # Maximum number of items in the buffer
        self.buffer_capacity = buffer_capacity

        # How many elements have been inserted in the buffer
        self.current_size = 0

        # Index of last inserted element
        self.current_idx = -1

        # Data buffers
        self.state_buffer = np.zeros(
            [self.buffer_capacity] + list(observation_space.shape),
            dtype=observation_space.dtype
        )

        self.action_buffer = np.zeros([self.buffer_capacity] + list(action_space.shape), dtype=action_space.dtype)
        self.reward_buffer = np.zeros([self.buffer_capacity], dtype=np.float32)
        self.dones_buffer = np.zeros([self.buffer_capacity], dtype=bool)

        self.extra_data = {} if extra_data is None else extra_data

        # Just a sentinel to simplify further calculations
        self.dones_buffer[self.current_idx] = True

    def store_transition(self, frame, action, reward, done, extra_info=None):
        """ Store given transition in the backend """
        self.current_idx = (self.current_idx + 1) % self.buffer_capacity

        self.state_buffer[self.current_idx] = frame
        self.action_buffer[self.current_idx] = action
        self.reward_buffer[self.current_idx] = reward
        self.dones_buffer[self.current_idx] = done

        for name in self.extra_data:
            self.extra_data[name][self.current_idx] = extra_info[name]

        if self.current_size < self.buffer_capacity:
            self.current_size += 1

        return self.current_idx

    def get_frame(self, idx, history_length=1):
        """ Return frame from the buffer """
        if idx >= self.current_size:
            raise VelException("Requested frame beyond the size of the buffer")

        if history_length > 1:
            assert self.state_buffer.shape[-1] == 1, \
                "State buffer must have last dimension of 1 if we want frame history"

        accumulator = []

        last_frame = self.state_buffer[idx]
        accumulator.append(last_frame)

        for i in range(history_length - 1):
            prev_idx = (idx - 1) % self.buffer_capacity

            if prev_idx == self.current_idx:
                raise VelException("Cannot provide enough history for the frame")
            elif self.dones_buffer[prev_idx]:
                # If previous frame was done - just append zeroes
                accumulator.append(np.zeros_like(last_frame))
            else:
                idx = prev_idx
                accumulator.append(self.state_buffer[idx])

        # We're pushing the elements in reverse order
        return np.concatenate(accumulator[::-1], axis=-1)

    def get_transition(self, frame_idx, history_length=1):
        """ Single transition with given index """
        past_frame, future_frame = self.get_frame_with_future(frame_idx, history_length)

        data_dict = {
            'observations': past_frame,
            'observations_next': future_frame,
            'actions': self.action_buffer[frame_idx],
            'rewards': self.reward_buffer[frame_idx],
            'dones': self.dones_buffer[frame_idx],
        }

        for name in self.extra_data:
            data_dict[name] = self.extra_data[name][frame_idx]

        return data_dict

    def get_frame_with_future(self, frame_idx, history_length=1):
        """ Return frame from the buffer together with the next frame """
        if frame_idx == self.current_idx:
            raise VelException("Cannot provide enough future for the frame")

        past_frame = self.get_frame(frame_idx, history_length)

        if history_length > 1:
            assert self.state_buffer.shape[-1] == 1, \
                "State buffer must have last dimension of 1 if we want frame history"

        if not self.dones_buffer[frame_idx]:
            next_idx = (frame_idx + 1) % self.buffer_capacity
            next_frame = self.state_buffer[next_idx]
        else:
            next_idx = (frame_idx + 1) % self.buffer_capacity
            next_frame = np.zeros_like(self.state_buffer[next_idx])

        if history_length > 1:
            future_frame = np.concatenate([
                past_frame.take(indices=np.arange(1, past_frame.shape[-1]), axis=-1), next_frame
            ], axis=-1)
        else:
            future_frame = next_frame

        return past_frame, future_frame

    def get_transitions(self, indexes, history_length=1):
        """ Return batch with given indexes """
        frame_batch_shape = (
                [indexes.shape[0]]
                + list(self.state_buffer.shape[1:-1])
                + [self.state_buffer.shape[-1] * history_length]
        )

        past_frame_buffer = np.zeros(frame_batch_shape, dtype=self.state_buffer.dtype)
        future_frame_buffer = np.zeros(frame_batch_shape, dtype=self.state_buffer.dtype)

        for buffer_idx, frame_idx in enumerate(indexes):
            past_frame_buffer[buffer_idx], future_frame_buffer[buffer_idx] = self.get_frame_with_future(
                frame_idx, history_length
            )

        actions = self.action_buffer[indexes]
        rewards = self.reward_buffer[indexes]
        dones = self.dones_buffer[indexes]

        data_dict = {
            'observations': past_frame_buffer,
            'actions': actions,
            'rewards': rewards,
            'observations_next': future_frame_buffer,
            'dones': dones,
        }

        for name in self.extra_data:
            data_dict[name] = self.extra_data[name][indexes]

        return data_dict

    def get_trajectories(self, index, rollout_length, history_length):
        """ Return batch consisting of *consecutive* transitions """
        indexes = np.arange(index - rollout_length + 1, index + 1, dtype=int)
        return self.get_transitions(indexes, history_length)

    def sample_batch_transitions(self, batch_size, history_length):
        """ Return indexes of next sample"""
        # Sample from up to total size
        if self.current_size < self.buffer_capacity:
            # -1 because we cannot take the last one
            return np.random.choice(self.current_size - 1, batch_size, replace=False)
        else:
            candidate = np.random.choice(self.buffer_capacity, batch_size, replace=False)

            forbidden_ones = (
                    np.arange(self.current_idx, self.current_idx + history_length)
                    % self.buffer_capacity
            )

            # Exclude these frames for learning as they may have some part of history overwritten
            while any(x in candidate for x in forbidden_ones):
                candidate = np.random.choice(self.buffer_capacity, batch_size, replace=False)

            return candidate

    def sample_batch_trajectories(self, rollout_length, history_length):
        """ Return indexes of next sample """
        # Sample from up to total size
        if self.current_size < self.buffer_capacity:
            if rollout_length + 1 > self.current_size:
                raise VelException("Not enough elements in the buffer to sample the rollout")

            # -1 because we cannot take the last one
            return np.random.choice(self.current_size - rollout_length) + rollout_length - 1
        else:
            if rollout_length + history_length > self.current_size:
                raise VelException("Not enough elements in the buffer to sample the rollout")

            candidate = np.random.choice(self.buffer_capacity)

            # These are the elements we cannot draw, as then we don't have enough history
            forbidden_ones = (
                    np.arange(self.current_idx, self.current_idx + history_length + rollout_length - 1)
                    % self.buffer_capacity
            )

            # Exclude these frames for learning as they may have some part of history overwritten
            while candidate in forbidden_ones:
                candidate = np.random.choice(self.buffer_capacity)

            return candidate
