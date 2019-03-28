import gym
import os.path

from gym.envs.registration import EnvSpec


from vel.openai.baselines import logger
from vel.openai.baselines.bench import Monitor
from vel.openai.baselines.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, FireResetEnv, EpisodicLifeEnv, WarpFrame, ClipRewardEnv,
    ScaledFloatFrame, FrameStack, FireEpisodicLifeEnv
)

from vel.rl.api import EnvFactory
from vel.rl.env.wrappers.clip_episode_length import ClipEpisodeLengthWrapper
from vel.util.situational import process_environment_settings


DEFAULT_SETTINGS = {
    'default': {
        'disable_reward_clipping': False,
        'disable_episodic_life': False,
        'monitor': False,
        'allow_early_resets': False,
        'scale_float_frames': False,
        'max_episode_frames': 10000,
        'frame_stack': None
    },
    'record': {
        'disable_reward_clipping': False,
        'disable_episodic_life': True,
        'monitor': False,
        'allow_early_resets': True,
        'scale_float_frames': False,
        'max_episode_frames': 10000,
        'frame_stack': None
    },
}


def env_maker(environment_id):
    """ Create a relatively raw atari environment """
    env = gym.make(environment_id)
    assert 'NoFrameskip' in env.spec.id

    # Wait for between 1 and 30 rounds doing nothing on start
    env = NoopResetEnv(env, noop_max=30)

    # Do the same action for k steps. Return max of last 2 frames. Return sum of rewards
    env = MaxAndSkipEnv(env, skip=4)

    return env


def wrapped_env_maker(environment_id, seed, serial_id, disable_reward_clipping=False, disable_episodic_life=False,
                      monitor=False, allow_early_resets=False, scale_float_frames=False,
                      max_episode_frames=10000, frame_stack=None):
    """ Wrap atari environment so that it's nicer to learn RL algorithms """
    env = env_maker(environment_id)
    env.seed(seed + serial_id)

    if max_episode_frames is not None:
        env = ClipEpisodeLengthWrapper(env, max_episode_length=max_episode_frames)

    # Monitoring the env
    if monitor:
        logdir = logger.get_dir() and os.path.join(logger.get_dir(), str(serial_id))
    else:
        logdir = None

    env = Monitor(env, logdir, allow_early_resets=allow_early_resets)

    if not disable_episodic_life:
        # Make end-of-life == end-of-episode, but only reset on true game over.
        # Done by DeepMind for the DQN and co. since it helps value estimation.
        env = EpisodicLifeEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        # Take action on reset for environments that are fixed until firing.
        if disable_episodic_life:
            env = FireEpisodicLifeEnv(env)
        else:
            env = FireResetEnv(env)

    # Warp frames to 84x84 as done in the Nature paper and later work.
    env = WarpFrame(env)

    if scale_float_frames:
        env = ScaledFloatFrame(env)

    if not disable_reward_clipping:
        # Bin reward to {+1, 0, -1} by its sign.
        env = ClipRewardEnv(env)

    if frame_stack is not None:
        env = FrameStack(env, frame_stack)

    return env


class ClassicAtariEnv(EnvFactory):
    """ Atari game environment wrapped in the same way as Deep Mind and OpenAI baselines """
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
        return wrapped_env_maker(self.envname, seed, serial_id, **settings)


def create(game, settings=None, presets=None):
    """ Vel factory function """
    return ClassicAtariEnv(game, settings, presets)
