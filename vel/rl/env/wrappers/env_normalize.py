import gym
import numpy as np

from vel.openai.baselines.common.running_mean_std import RunningMeanStd


class EnvNormalize(gym.Wrapper):
    """
    Single environment normalization based on VecNormalize from OpenAI baselines
    """
    def __init__(self, env, normalize_observations=True, normalize_returns=True,
                 clip_observations=10., clip_rewards=10., gamma=0.99, epsilon=1e-8):
        super().__init__(env)

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if normalize_observations else None
        self.ret_rms = RunningMeanStd(shape=()) if normalize_returns else None
        self.clipob = clip_observations
        self.cliprew = clip_rewards
        self.ret = 0.0
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.env.step(action)

        self.ret = self.ret * self.gamma + rews

        obs = self._filter_observation(obs)

        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret]))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        return obs, rews, news, infos

    def _filter_observation(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs[None])
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)

            return obs.astype(np.float32)
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.env.reset()
        return self._filter_observation(obs)
