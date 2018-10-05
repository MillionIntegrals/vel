import typing
import gym

from vel.api import BatchInfo
from vel.api.base import Model
from vel.openai.baselines.common.vec_env import VecEnv


class EnvRollerBase:
    """ Class generating environment rollouts """

    @property
    def environment(self) -> typing.Union[gym.Env, VecEnv]:
        """ Reference to environment being evaluated """
        raise NotImplementedError

    def rollout(self, batch_info: BatchInfo, model: Model) -> dict:
        """ Roll-out the environment and return it """
        raise NotImplementedError

    def metrics(self) -> list:
        """ List of metrics to track for this learning process """
        return []


# noinspection PyAbstractClass
class ReplayEnvRollerBase(EnvRollerBase):
    """ Class generating environment rollouts with experience replay """

    def sample(self, batch_info: BatchInfo, model: Model) -> dict:
        """ Sample experience from replay buffer and return a batch """
        raise NotImplementedError

    def is_ready_for_sampling(self) -> bool:
        """ If buffer is ready for drawing samples from it (usually checks if there is enough data) """
        raise NotImplementedError

    def update(self, sample, batch_info):
        """ Perform update of the internal state of the buffer - e.g. for the prioritized replay weights """
        pass


class EnvRollerFactory:
    """ Factory for env rollers """

    def instantiate(self, environment, device, settings) -> EnvRollerBase:
        """ Instantiate env roller """
        raise NotImplementedError


class ReplayEnvRollerFactory(EnvRollerFactory):
    """ Factory for env rollers """

    def instantiate(self, environment, device, settings) -> ReplayEnvRollerBase:
        """ Instantiate env roller """
        raise NotImplementedError
