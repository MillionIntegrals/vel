import gym
import os.path

from gym.envs.registration import EnvSpec

from vel.openai.baselines import logger
from vel.openai.baselines.bench import Monitor
from vel.rl.api import EnvFactory
from vel.rl.env.wrappers.env_normalize import EnvNormalize
from vel.util.situational import process_environment_settings


DEFAULT_SETTINGS = {
    'default': {
        'monitor': False,
        'allow_early_resets': False,
        'normalize_observations': False,
        'normalize_returns': False,
    },
    'record': {
        'monitor': False,
        'allow_early_resets': True,
        'normalize_observations': False,
        'normalize_returns': False,
    }
}


def env_maker(environment_id, seed, serial_id, monitor=False, allow_early_resets=False, normalize_observations=False,
              normalize_returns=False, normalize_gamma=0.99):
    """ Create a relatively raw atari environment """
    env = gym.make(environment_id)
    env.seed(seed + serial_id)

    # Monitoring the env
    if monitor:
        logdir = logger.get_dir() and os.path.join(logger.get_dir(), str(serial_id))
    else:
        logdir = None

    env = Monitor(env, logdir, allow_early_resets=allow_early_resets)

    if normalize_observations or normalize_returns:
        env = EnvNormalize(
            env,
            normalize_observations=normalize_observations,
            normalize_returns=normalize_returns,
            gamma=normalize_gamma
        )

    return env


class MujocoEnv(EnvFactory):
    """ Atari game environment wrapped in the same way as Deep Mind and OpenAI baselines """
    def __init__(self, envname, normalize_observations=False, normalize_returns=False, settings=None, presets=None):
        self.envname = envname

        settings = settings if settings is not None else {}

        if normalize_observations:
            settings['normalize_observations'] = True

        if normalize_returns:
            settings['normalize_returns'] = True

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


def create(game, normalize_returns=False, settings=None, presets=None):
    """ Vel factory function """
    return MujocoEnv(
        envname=game,
        normalize_returns=normalize_returns,
        settings=settings,
        presets=presets
    )
