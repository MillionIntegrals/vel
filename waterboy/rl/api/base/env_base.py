from gym import Env
from gym.envs.registration import EnvSpec
from waterboy.openai.baselines.common.vec_env import VecEnv


class EnvFactoryBase:
    """ Base class for environment factory """

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        raise NotImplementedError

    def instantiate(self, seed=0, serial_id=1) -> Env:
        """ Create a new Env instance """
        raise NotImplementedError


class VecEnvFactoryBase:
    """ Base class for vector environment factory """

    def instantiate(self, parallel_envs, seed=0) -> VecEnv:
        """ Create a new VecEnv instance """
        raise NotImplementedError

