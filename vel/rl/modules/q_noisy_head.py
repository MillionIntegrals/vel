import torch.nn as nn

import gym.spaces as spaces

from vel.rl.modules.noisy_linear import NoisyLinear


class QNoisyHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """
    def __init__(self, input_dim, action_space, initial_std_dev=0.4, factorized_noise=True):
        super().__init__()

        # Q-function requires a discrete action space
        assert isinstance(action_space, spaces.Discrete)

        self.action_space = action_space

        self.linear_layer = NoisyLinear(
            input_dim, action_space.n, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise
        )

    def reset_weights(self):
        self.linear_layer.reset_weights()

    def forward(self, input_data):
        return self.linear_layer(input_data)

    def sample(self, q_values):
        """ Sample from epsilon-greedy strategy with given q-values """
        return q_values.argmax(dim=1)
