import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import gym.spaces as spaces


class QHead(nn.Module):
    """ Network head calculating Q-function value for each action. """
    def __init__(self, input_dim, action_space):
        super().__init__()

        # For now let's fix discrete action space, I'll generalize it later
        assert isinstance(action_space, spaces.Discrete)

        self.linear_layer = nn.Linear(input_dim, action_space.n)
        self.action_space = action_space

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=1.0)
                init.constant_(m.bias, 0.0)

    def forward(self, input_data):
        return self.linear_layer(input_data)

    def sample(self, q_values, epsilon):
        """ Sample from epsilon-greedy strategy with given q-values """
        policy_samples = q_values.argmax(dim=1)
        random_samples = torch.randint_like(policy_samples, self.action_space.n)
        selector = torch.rand(random_samples.shape, device=q_values.device)
        return torch.where(selector > epsilon, policy_samples, random_samples)
