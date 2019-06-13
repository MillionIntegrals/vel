import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import gym.spaces as spaces


class QDistributionalHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """
    def __init__(self, input_dim, action_space, vmin: float, vmax: float, atoms: int = 1):
        super().__init__()

        # Q-function requires a discrete action space
        assert isinstance(action_space, spaces.Discrete)
        assert vmax > vmin

        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax

        self.action_space = action_space
        self.action_size = action_space.n

        self.atom_delta = (self.vmax - self.vmin) / (self.atoms - 1)

        self.linear_layer = nn.Linear(input_dim, self.action_size * self.atoms)

        self.register_buffer('support_atoms', torch.linspace(self.vmin, self.vmax, self.atoms))

    def histogram_info(self) -> dict:
        """ Return extra information about histogram """
        return {
            'support_atoms': self.support_atoms,
            'atom_delta': self.atom_delta,
            'vmin': self.vmin,
            'vmax': self.vmax,
            'num_atoms': self.atoms
        }

    def reset_weights(self):
        init.orthogonal_(self.linear_layer.weight, gain=1.0)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data):
        histogram_logits = self.linear_layer(input_data).view(input_data.size(0), self.action_size, self.atoms)
        histogram_log = F.log_softmax(histogram_logits, dim=2)

        # Calculate log-softmax to establish log-probability distribution
        return histogram_log

    def sample(self, histogram_logits):
        """ Sample from a greedy strategy with given q-value histogram """
        histogram_probs = histogram_logits.exp()  # Batch size * actions * atoms
        atoms = self.support_atoms.view(1, 1, self.atoms)  # Need to introduce two new dimensions
        return (histogram_probs * atoms).sum(dim=-1).argmax(dim=1)  # Argmax of expectation
