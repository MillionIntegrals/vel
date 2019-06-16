import typing

from vel.api import BatchInfo
from vel.openai.baselines.common.vec_env import VecEnv
from vel.rl.api.rollout import Rollout


class EnvRollerBase:
    """ Class generating environment rollouts """

    @property
    def environment(self) -> VecEnv:
        """ Reference to environment being evaluated """
        raise NotImplementedError

    def rollout(self, batch_info: BatchInfo, number_of_steps: int) -> Rollout:
        """ Roll-out the environment and return it """
        raise NotImplementedError

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []


# noinspection PyAbstractClass
class ReplayEnvRollerBase(EnvRollerBase):
    """ Class generating environment rollouts with experience replay """

    def sample(self, batch_info: BatchInfo, number_of_steps: int) -> Rollout:
        """ Sample experience from replay buffer and return a batch """
        raise NotImplementedError

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        raise NotImplementedError

    def initial_memory_size_hint(self) -> typing.Optional[int]:
        """ Hint how much data is needed to begin sampling, required only for diagnostics """
        return None

    def update(self, rollout, batch_info):
        """ Perform update of the internal state of the buffer - e.g. for the prioritized replay weights """
        pass


class EnvRollerFactoryBase:
    """ Factory for env rollers """

    def instantiate(self, environment, policy, device) -> EnvRollerBase:
        """ Instantiate env roller """
        raise NotImplementedError


class ReplayEnvRollerFactoryBase(EnvRollerFactoryBase):
    """ Factory for env rollers """

    def instantiate(self, environment, policy, device) -> ReplayEnvRollerBase:
        """ Instantiate env roller """
        raise NotImplementedError
