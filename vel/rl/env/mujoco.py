import gym
import os.path

from gym.envs.registration import EnvSpec

from vel.rl.api.base import EnvFactory
from vel.openai.baselines import logger
from vel.openai.baselines.bench import Monitor
from vel.rl.env.wrappers.env_normalize import EnvNormalize


DEFAULT_SETTINGS = {
    'default': {
        'monitor': False,
        'allow_early_resets': False,
        'normalize_observations': False,
        'normalize_returns': False,
    },
    'raw': {
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
    def __init__(self, envname, env_settings=None, normalize_observations=False, normalize_returns=False):
        self.envname = envname

        env_settings = env_settings if env_settings is not None else {}
        env_keys = set(DEFAULT_SETTINGS.keys()).union(set(env_settings.keys()))

        self.presets = {}

        for key in env_keys:
            preset_settings = DEFAULT_SETTINGS.get(key, {}).copy()

            if normalize_observations:
                preset_settings['normalize_observations'] = True

            if normalize_returns:
                preset_settings['normalize_returns'] = True

            if key in env_settings:
                preset_settings.update(env_settings[key])

            self.presets[key] = preset_settings

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        return gym.spec(self.envname)

    def get_preset(self, preset_key='default') -> dict:
        """ Get env settings for given preset """
        return self.presets[preset_key]

    def instantiate(self, seed=0, serial_id=0, preset='default', extra_args=None) -> gym.Env:
        """ Make a single environment compatible with the experiments """
        settings = self.get_preset(preset)
        return env_maker(self.envname, seed, serial_id, **settings)


def create(game, env_settings=None, normalize_observations=False, normalize_returns=False):
    return MujocoEnv(
        envname=game,
        env_settings=env_settings,
        normalize_observations=normalize_observations,
        normalize_returns=normalize_returns
    )
