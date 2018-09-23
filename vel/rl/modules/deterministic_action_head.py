import numpy as np
import gym.spaces as spaces

import torch
import torch.nn as nn
import torch.nn.init as init


class DeterministicActionHead(nn.Module):
    """
    Network head for action determination. Returns deterministic action depending on the inputs
    """

    def __init__(self, input_dim, action_space):
        super().__init__()

        self.action_space = action_space

        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1

        assert (np.abs(action_space.low) == action_space.high).all()  # we assume symmetric actions.
        self.register_buffer('max_action', torch.from_numpy(action_space.high))

        self.linear_layer = nn.Linear(input_dim, action_space.shape[0])

    def forward(self, input_data):
        return torch.tanh(self.linear_layer(input_data)) * self.max_action

    def sample(self, params, **_):
        """ Sample from a probability space of all actions """
        return {
            'actions': self(params)
        }

    def reset_weights(self):
        """ Initialize weights to sane defaults """
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)
