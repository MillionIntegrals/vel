import gym
import numpy as np

from vel.exceptions import VelException


class DequeBufferBackend:
    """ Simple backend behind DequeBuffer """

    def __init__(self, buffer_capacity: int, observation_space: gym.Space, action_space: gym.Space):
        # Maximum number of items in the buffer
        self.buffer_capacity = buffer_capacity

        # How many elements have been inserted in the buffer
        self.current_size = 0

        # Index of last inserted element
        self.current_idx = -1

        # Data buffers
        self.frame_buffer = np.zeros(
            [self.buffer_capacity] + list(observation_space.shape),
            dtype=observation_space.dtype
        )
        self.action_buffer = np.zeros([self.buffer_capacity], dtype=action_space.dtype)
        self.reward_buffer = np.zeros([self.buffer_capacity], dtype=float)
        self.dones_buffer = np.zeros([self.buffer_capacity], dtype=bool)
        self.prob_buffer = np.zeros([self.buffer_capacity], dtype=float)

        # Just a sentinel to simplify further calculations
        self.dones_buffer[self.current_idx] = True

    def store_transition(self, frame, action, reward, done, prob=0.0):
        """ Store given transition in the backend """
        self.current_idx = (self.current_idx + 1) % self.buffer_capacity

        self.frame_buffer[self.current_idx] = frame
        self.action_buffer[self.current_idx] = action
        self.reward_buffer[self.current_idx] = reward
        self.dones_buffer[self.current_idx] = done
        self.prob_buffer[self.current_idx] = prob

        if self.current_size < self.buffer_capacity:
            self.current_size += 1

        return self.current_idx

    def get_frame(self, idx, history):
        """ Return frame from the buffer """
        if idx >= self.current_size:
            raise VelException("Requested frame beyond the size of the buffer")

        accumulator = []

        last_frame = self.frame_buffer[idx]
        accumulator.append(last_frame)

        for i in range(history - 1):
            prev_idx = (idx - 1) % self.buffer_capacity

            if prev_idx == self.current_idx:
                raise VelException("Cannot provide enough history for the frame")
            elif self.dones_buffer[prev_idx]:
                # If previous frame was done - just append zeroes
                accumulator.append(np.zeros_like(last_frame))
            else:
                idx = prev_idx
                accumulator.append(self.frame_buffer[idx])

        # We're pushing the elements in reverse order
        return np.concatenate(accumulator[::-1], axis=-1)

    def get_frame_with_future(self, idx, history):
        """ Return frame from the buffer together with the next frame """
        if idx == self.current_idx:
            raise VelException("Cannot provide enough future for the frame")

        past_frame = self.get_frame(idx, history)

        future_frame = np.zeros_like(past_frame)

        future_frame[:, :, :-1] = past_frame[:, :, 1:]

        if not self.dones_buffer[idx]:
            next_idx = (idx + 1) % self.buffer_capacity
            next_frame = self.frame_buffer[next_idx]
            future_frame[:, :, -1:] = next_frame

        return past_frame, future_frame

    def get_batch(self, indexes, history):
        """ Return batch with given indexes """
        frame_batch_shape = (
                [indexes.shape[0]] + list(self.frame_buffer.shape[1:-1]) + [self.frame_buffer.shape[-1] * history]
        )

        past_frame_buffer = np.zeros(frame_batch_shape, dtype=np.uint8)
        future_frame_buffer = np.zeros(frame_batch_shape, dtype=np.uint8)

        for buffer_idx, frame_idx in enumerate(indexes):
            past_frame_buffer[buffer_idx], future_frame_buffer[buffer_idx] = self.get_frame_with_future(
                frame_idx, history
            )

        actions = self.action_buffer[indexes]
        rewards = self.reward_buffer[indexes]
        dones = self.dones_buffer[indexes]

        return past_frame_buffer, actions, rewards, future_frame_buffer, dones

    def sample_batch_uniform(self, batch_size, history):
        """ Return indexes of next sample"""
        # Sample from up to total size
        if self.current_size < self.buffer_capacity:
            # -1 because we cannot take the last one
            return np.random.choice(self.current_size - 1, batch_size, replace=False)
        else:
            candidate = np.random.choice(self.buffer_capacity, batch_size, replace=False)

            forbidden_ones = (
                    np.arange(self.current_idx, self.current_idx + history)
                    % self.buffer_capacity
            )

            # Exclude these frames for learning as they may have some part of history overwritten
            while any(x in candidate for x in forbidden_ones):
                candidate = np.random.choice(self.buffer_capacity, batch_size, replace=False)

            return candidate
