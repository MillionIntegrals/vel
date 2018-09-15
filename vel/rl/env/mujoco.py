import gym
import os.path

from gym.envs.registration import EnvSpec

from vel.rl.api.base import EnvFactory
from vel.openai.baselines import logger
from vel.openai.baselines.bench import Monitor


DEFAULT_SETTINGS = {
    'default': {
        'monitor': False,
        'allow_early_resets': False
    },
    'raw': {
        'monitor': False,
        'allow_early_resets': True
    }
}


def env_maker(environment_id, seed, serial_id, monitor=False, allow_early_resets=False):
    """ Create a relatively raw atari environment """
    env = gym.make(environment_id)
    env.seed(seed + serial_id)

    # Monitoring the env
    if monitor:
        logdir = logger.get_dir() and os.path.join(logger.get_dir(), str(serial_id)),
    else:
        logdir = None

    env = Monitor(env, logdir, allow_early_resets=allow_early_resets)

    return env


class MujocoEnv(EnvFactory):
    """ Atari game environment wrapped in the same way as Deep Mind and OpenAI baselines """
    def __init__(self, envname, env_settings=None):
        self.envname = envname

        env_settings = env_settings if env_settings is not None else {}
        env_keys = set(DEFAULT_SETTINGS.keys()).union(set(env_settings.keys()))

        self.presets = {}

        for key in env_keys:
            self.presets[key] = DEFAULT_SETTINGS.get(key, {})
            self.presets[key].update(env_settings.get(key, {}))

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        return gym.spec(self.envname)

    def instantiate(self, seed=0, serial_id=0, preset='default') -> gym.Env:
        """ Make a single environment compatible with the experiments """
        settings = self.presets[preset]
        return env_maker(self.envname, seed, serial_id, **settings)


def create(game, env_settings=None):
    return MujocoEnv(game, env_settings)
