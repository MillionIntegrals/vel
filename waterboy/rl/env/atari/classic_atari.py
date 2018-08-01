import gym
import os.path

from gym.envs.registration import EnvSpec

from waterboy.rl.api.base import EnvFactoryBase

from waterboy.openai.baselines.bench import Monitor
from waterboy.openai.baselines import logger

from waterboy.openai.baselines.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, FireResetEnv, EpisodicLifeEnv, WarpFrame, ClipRewardEnv
)


def env_maker(environment_id):
    """ Create a relatively raw atari environment """
    env = gym.make(environment_id)
    assert 'NoFrameskip' in env.spec.id

    # Wait for between 1 and 30 rounds doing nothing on start
    env = NoopResetEnv(env, noop_max=30)

    # Do the same action for k steps. Return max of last 2 frames. Return sum of rewards
    env = MaxAndSkipEnv(env, skip=4)

    return env


def wrapped_env_maker(environment_id, seed, serial_id, disable_reward_clipping=False, disable_episodic_life=False, monitor=True):
    """ Wrap atari environment so that it's nicer to learn RL algorithms """
    env = env_maker(environment_id)
    env.seed(seed + serial_id)

    # Monitoring the env to measure sth (?)
    if monitor:
        env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(serial_id)))

    if not disable_episodic_life:
        # Make end-of-life == end-of-episode, but only reset on true game over.
        # Done by DeepMind for the DQN and co. since it helps value estimation.
        env = EpisodicLifeEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        # Take action on reset for environments that are fixed until firing.
        env = FireResetEnv(env)

    # Warp frames to 84x84 as done in the Nature paper and later work.
    env = WarpFrame(env)

    if not disable_reward_clipping:
        # Bin reward to {+1, 0, -1} by its sign.
        env = ClipRewardEnv(env)

    return env


def raw_env_maker(environment_id, seed, serial_id, disable_reward_clipping=False):
    """ Wrap atari environment so that it's nicer to learn RL algorithms """
    env = env_maker(environment_id)
    env.seed(seed + serial_id)

    # Monitoring the env to measure sth (?)
    env = Monitor(env, None)

    # if not disable_episodic_life:
    #     # Make end-of-life == end-of-episode, but only reset on true game over.
    #     # Done by DeepMind for the DQN and co. since it helps value estimation.
    #     env = EpisodicLifeEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        # Take action on reset for environments that are fixed until firing.
        env = FireResetEnv(env)

    # Warp frames to 84x84 as done in the Nature paper and later work.
    env = WarpFrame(env)

    if not disable_reward_clipping:
        # Bin reward to {+1, 0, -1} by its sign.
        env = ClipRewardEnv(env)

    return env


class ClassicAtariEnv(EnvFactoryBase):
    """ Atari game environment wrapped in the same way as Deep Mind and OpenAI baselines """
    def __init__(self, envname, disable_reward_clipping=False, disable_episodic_life=False, monitor=True):
        self.envname = envname
        self.disable_reward_clipping = disable_reward_clipping
        self.disable_episodic_life = disable_episodic_life
        self.monitor = monitor

    def specification(self) -> EnvSpec:
        """ Return environment specification """
        return gym.spec(self.envname)

    def instantiate(self, seed=0, serial_id=0, raw=False) -> gym.Env:
        """ Make a single environment compatible with the experiments """
        if raw:
            return raw_env_maker(
                self.envname, seed, serial_id,
                disable_reward_clipping=self.disable_reward_clipping,
            )
        else:
            return wrapped_env_maker(
                self.envname, seed, serial_id,
                disable_reward_clipping=self.disable_reward_clipping,
                disable_episodic_life=self.disable_episodic_life,
                monitor=self.monitor
            )


def create(game, model_config, disable_reward_clipping=False, monitor=True):
    logger.configure(dir=model_config.openai_dir())
    return ClassicAtariEnv(
        game, disable_reward_clipping=disable_reward_clipping, monitor=monitor
    )
