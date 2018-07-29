from waterboy.openai.baselines.common.vec_env import VecEnv
from waterboy.openai.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from waterboy.openai.baselines.common.vec_env.vec_frame_stack import VecFrameStack


class SubprocVecEnvWrapper:
    """ Wrapper for an environment to create sub-process vector environment """

    def __init__(self, env, frame_history):
        self.env = env
        self.frame_history = frame_history

    def instantiate(self, parallel_envs, seed=0) -> VecEnv:
        """ Make parallel environments """
        envs = SubprocVecEnv([self._creation_function(i, seed) for i in range(parallel_envs)])
        envs = VecFrameStack(envs, self.frame_history)

        return envs

    def _creation_function(self, idx, seed):
        """ Helper function to create a proper closure around supplied values """
        return lambda: self.env.instantiate(seed=seed, serial_id=idx)


def create(env, frame_history):
    return SubprocVecEnvWrapper(env, frame_history=frame_history)
