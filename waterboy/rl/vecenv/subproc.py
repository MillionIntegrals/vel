from waterboy.openai.baselines.common.vec_env import VecEnv
from waterboy.openai.baselines.common.atari_wrappers import FrameStack
from waterboy.openai.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from waterboy.openai.baselines.common.vec_env.vec_frame_stack import VecFrameStack
from waterboy.rl.api.base import VecEnvFactoryBase


class SubprocVecEnvWrapper(VecEnvFactoryBase):
    """ Wrapper for an environment to create sub-process vector environment """

    def __init__(self, env, frame_history):
        self.env = env
        self.frame_history = frame_history

    def instantiate(self, parallel_envs, seed=0, raw=False) -> VecEnv:
        """ Make parallel environments """
        envs = SubprocVecEnv([self._creation_function(i, seed, raw) for i in range(parallel_envs)])
        envs = VecFrameStack(envs, self.frame_history)

        return envs

    def instantiate_single(self, seed=0, raw=False):
        """ Create a new VecEnv instance - single """
        env = self.env.instantiate(seed=seed, serial_id=0, raw=raw)
        return FrameStack(env, self.frame_history)

    def _creation_function(self, idx, seed, raw):
        """ Helper function to create a proper closure around supplied values """
        return lambda: self.env.instantiate(seed=seed, serial_id=idx, raw=raw)


def create(env, frame_history):
    return SubprocVecEnvWrapper(env, frame_history=frame_history)
