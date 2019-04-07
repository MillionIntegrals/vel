import typing

from vel.openai.baselines.common.vec_env import VecEnv

from .rollout import Trajectories, Transitions


class ReplayBuffer:
    """ Base class for a replay buffer """
    def __init__(self):
        pass

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        raise NotImplementedError

    def initial_memory_size_hint(self) -> typing.Optional[int]:
        """ Hint how much data is needed to begin sampling, required only for diagnostics """
        return None

    def update(self, rollout, batch_info):
        """ Perform update of the internal state of the buffer - e.g. for the prioritized replay weights """
        raise NotImplementedError

    def sample_transitions(self, batch_size, batch_info) -> Transitions:
        """ Sample transitions from replay buffer """
        raise NotImplementedError

    def sample_forward_transitions(self, batch_size, batch_info, forward_steps: int,
                                   discount_factor: float) -> Transitions:
        """
        Sample transitions from replay buffer with _forward steps_.
        That is, instead of getting a transition s_t -> s_t+1 with reward r,
        get a transition s_t -> s_t+n with sum of intermediate rewards.

        Used in a variant of Deep Q-Learning
        """
        raise NotImplementedError

    def sample_trajectories(self, rollout_length, batch_info) -> Trajectories:
        """ Sample transitions from replay buffer """
        raise NotImplementedError

    def store_transition(self, frame, action, reward, done, extra_info=None):
        """ Store given transition in the backend """
        raise NotImplementedError


class ReplayBufferFactory:
    """ Create a replay buffer based on supplied environment """

    def instantiate(self, environment: VecEnv):
        raise NotImplementedError
