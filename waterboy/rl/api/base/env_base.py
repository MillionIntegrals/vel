from gym import Env
from gym.envs.registration import EnvSpec
from waterboy.openai.baselines.common.vec_env import VecEnv


class EnvFactoryBase:
    """ Base class for environment factory """

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        raise NotImplementedError

    def instantiate(self, seed=0, serial_id=1, raw=False) -> Env:
        """ Create a new Env instance """
        raise NotImplementedError


class VecEnvFactoryBase:
    """ Base class for vector environment factory """

    def instantiate(self, parallel_envs, seed=0, raw=False) -> VecEnv:
        """ Create a new VecEnv instance """
        raise NotImplementedError

    def instantiate_single(self, seed=0, raw=False) -> VecEnv:
        """ Create a new VecEnv instance - single """
        raise NotImplementedError

