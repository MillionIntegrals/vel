import gym


class ClipEpisodeLengthWrapper(gym.Wrapper):
    """ Env wrapper that clips number of frames an episode can last """
    def __init__(self, env, max_episode_length):
        super().__init__(env)

        self.max_episode_length = max_episode_length
        self.current_episode_length = 0

    def reset(self, **kwargs):
        self.current_episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_episode_length += 1
        ob, reward, done, info = self.env.step(action)

        if self.current_episode_length > self.max_episode_length:
            done = True
            info['clipped_length'] = True

        return ob, reward, done, info
