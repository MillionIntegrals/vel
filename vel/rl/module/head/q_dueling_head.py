import torch.nn as nn
import torch.nn.init as init

import gym.spaces as spaces


class QDuelingHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action using two separate inputs. """
    def __init__(self, input_dim, action_space):
        super().__init__()

        # Q-function requires a discrete action space
        assert isinstance(action_space, spaces.Discrete)

        self.linear_layer_advantage = nn.Linear(input_dim, action_space.n)
        self.linear_layer_value = nn.Linear(input_dim, 1)
        self.action_space = action_space

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=1.0)
                init.constant_(m.bias, 0.0)

    def forward(self, advantage_data, value_data):
        adv = self.linear_layer_advantage(advantage_data)
        value = self.linear_layer_value(value_data)
        # Advantage must be 0-centered

        return (adv - adv.mean(dim=1, keepdim=True)) + value

    def sample(self, q_values):
        """ Sample from greedy strategy with given q-values """
        return q_values.argmax(dim=1)
