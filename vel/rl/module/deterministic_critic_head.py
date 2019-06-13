import torch.nn as nn
import torch.nn.init as init


class DeterministicCriticHead(nn.Module):
    """
    Network head for action-dependent critic.
    Returns deterministic action-value for given combination of action and state.
    """

    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, input_data):
        return self.linear(input_data)[:, 0]

    def reset_weights(self):
        """ Initialize weights to sane defaults """
        init.uniform_(self.linear.weight, -3e-3, 3e-3)
        init.zeros_(self.linear.bias)
