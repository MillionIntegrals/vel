import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import gym.spaces as spaces


class ActionHead(nn.Module):
    """
    Network head for action determination.
    Returns action logits.
    """

    def __init__(self, input_dim, action_space, argmax_sampling=False):
        super().__init__()

        # For now let's fix discrete action space, I'll generalize it later
        assert isinstance(action_space, spaces.Discrete)

        self.linear_layer = nn.Linear(input_dim, action_space.n)

        self.action_space = action_space
        self.argmax_sampling = argmax_sampling

    def forward(self, input_data):
        return F.log_softmax(self.linear_layer(input_data), dim=1)

    def sample(self, logits):
        """ Sample from a probability space of all actions """
        if self.argmax_sampling:
            return torch.argmax(logits, dim=-1)
        else:
            u = torch.rand_like(logits)
            return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=0.01)
                init.constant_(m.bias, 0.0)

    def entropy(self, logits):
        """ Categorical distribution entropy calculation - sum probs * log(probs) """
        probs = torch.exp(logits)
        entropy = - torch.sum(probs * logits, dim=-1)
        return entropy

