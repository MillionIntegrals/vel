import gym.spaces as spaces

import torch
import torch.nn as nn
import torch.nn.functional as F


from vel.rl.modules.noisy_linear import NoisyLinear


class QDistributionalNoisyDuelingHead(nn.Module):
    """ Network head calculating Q-function value for each (discrete) action. """
    def __init__(self, input_dim, action_space, vmin: float, vmax: float, atoms: int = 1,
                 initial_std_dev: float = 0.4, factorized_noise: bool = True):
        super().__init__()

        # Q-function requires a discrete action space
        assert isinstance(action_space, spaces.Discrete)
        assert vmax > vmin

        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax

        self.action_size = action_space.n
        self.action_space = action_space

        self.atom_delta = (self.vmax - self.vmin) / (self.atoms - 1)

        self.linear_layer_advantage = NoisyLinear(
            input_dim, self.action_size * self.atoms, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise
        )

        self.linear_layer_value = NoisyLinear(
            input_dim, self.atoms, initial_std_dev=initial_std_dev, factorized_noise=factorized_noise
        )

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
        self.linear_layer_advantage.reset_weights()
        self.linear_layer_value.reset_weights()

    def forward(self, advantage_features, value_features):
        adv = self.linear_layer_advantage(advantage_features).view(-1, self.action_size, self.atoms)
        val = self.linear_layer_value(value_features).view(-1, 1, self.atoms)

        # I'm quite unsure if this is the right way to combine these, but this is what paper seems to be suggesting
        # and I don't know any better way.
        histogram_output = val + adv - adv.mean(dim=1, keepdim=True)

        return F.log_softmax(histogram_output, dim=2)

    def sample(self, histogram_logits):
        """ Sample from a greedy strategy with given q-value histogram """
        histogram_probs = histogram_logits.exp()  # Batch size * actions * atoms
        atoms = self.support_atoms.view(1, 1, self.atoms)  # Need to introduce two new dimensions
        return (histogram_probs * atoms).sum(dim=-1).argmax(dim=1)  # Argmax of expectation
