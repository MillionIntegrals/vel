import torch
import torch.nn as nn
import torch.nn.init as init

import gym.spaces as spaces


class QHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """
    def __init__(self, input_dim, action_space):
        super().__init__()

        # Q-function requires a discrete action space
        assert isinstance(action_space, spaces.Discrete)

        self.linear_layer = nn.Linear(input_dim, action_space.n)
        self.action_space = action_space

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=1.0)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data):
        return self.linear_layer(input_data)

    def sample(self, q_values):
        """ Sample from epsilon-greedy strategy with given q-values """
        return q_values.argmax(dim=1)

