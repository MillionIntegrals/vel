import gym.spaces as spaces

import torch
import torch.nn as nn
import torch.nn.init as init

import vel.util.network as net_util


class DeterministicCriticHead(nn.Module):
    """
    Network head for action-dependent critic.
    Returns deterministic action-value for given combination of action and state.
    """

    def __init__(self, input_dim, action_space, hidden_dim=64, layer_norm=True, activation='relu'):
        super().__init__()

        self.action_space = action_space

        assert isinstance(action_space, spaces.Box)
        assert len(action_space.shape) == 1

        self.linear_layer = nn.Linear(input_dim + action_space.shape[0], hidden_dim)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

        self.activation = net_util.activation(activation)()

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, observation_data, action_data):
        combined_data = torch.cat([observation_data, action_data], dim=1)

        linear_output = self.linear_layer(combined_data)

        if self.layer_norm is not None:
            linear_output = self.layer_norm(linear_output)

        activated = self.activation(linear_output)

        final_output = self.output_layer(activated)

        return final_output[:, 0]

    def sample(self, params, **kwargs):
        """ Sample from a probability space of all actions """
        return self.head.sample(params, **kwargs)

    def reset_weights(self):
        """ Initialize weights to sane defaults """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=0.01)
                init.constant_(m.bias, 0.0)
