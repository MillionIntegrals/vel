import gym
import os.path

from gym.envs.registration import EnvSpec

from vel.openai.baselines import logger
from vel.openai.baselines.bench import Monitor

from vel.rl.api import EnvFactory
from vel.util.situational import process_environment_settings


DEFAULT_SETTINGS = {
    'default': {
        'monitor': False,
        'allow_early_resets': False,
    },
    'record': {
        'monitor': False,
        'allow_early_resets': True,
    }
}


def env_maker(environment_id, seed, serial_id, monitor=False, allow_early_resets=False):
    """ Create a classic control environment with basic set of wrappers """
    env = gym.make(environment_id)
    env.seed(seed + serial_id)

    # Monitoring the env
    if monitor:
        logdir = logger.get_dir() and os.path.join(logger.get_dir(), str(serial_id))
    else:
        logdir = None

    env = Monitor(env, logdir, allow_early_resets=allow_early_resets)

    return env


class ClassicControlEnv(EnvFactory):
    """ A set of "classic control" environments from OpenAI gym """

    def __init__(self, envname, settings=None, presets=None):
        self.envname = envname
        self.settings = process_environment_settings(DEFAULT_SETTINGS, settings, presets)

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        return gym.spec(self.envname)

    def get_preset(self, preset_key='default') -> dict:
        """ Get env settings for given preset """
        return self.settings[preset_key]

    def instantiate(self, seed=0, serial_id=0, preset='default', extra_args=None) -> gym.Env:
        """ Make a single environment compatible with the experiments """
        settings = self.get_preset(preset)
        return env_maker(self.envname, seed, serial_id, **settings)


def create(game, settings=None, presets=None):
    """ Vel factory function """
    return ClassicControlEnv(
        envname=game,
        settings=settings,
        presets=presets
    )
