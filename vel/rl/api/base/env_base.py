from gym import Env
from gym.envs.registration import EnvSpec
from vel.openai.baselines.common.vec_env import VecEnv


class EnvFactory:
    """ Base class for environment factory """

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        raise NotImplementedError

    def instantiate(self, seed=0, serial_id=1, preset='default', extra_args=None) -> Env:
        """ Create a new Env instance """
        raise NotImplementedError


class VecEnvFactory:
    """ Base class for vector environment factory """

    def instantiate(self, parallel_envs, seed=0, preset='default') -> VecEnv:
        """ Create a new VecEnv instance """
        raise NotImplementedError

    def instantiate_single(self, seed=0, preset='default') -> VecEnv:
        """ Create a new VecEnv instance - single """
        raise NotImplementedError

