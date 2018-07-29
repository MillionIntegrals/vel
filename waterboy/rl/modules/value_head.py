import torch.nn as nn


class ValueHead(nn.Module):
    """ Network head for value determination """
    def __init__(self, input_dim):
        super().__init__()

        self.linear_layer = nn.Linear(input_dim, 1)

    def forward(self, input_data):
        return self.linear_layer(input_data)[:, 0]
