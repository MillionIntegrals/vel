import torch.nn as nn
import torch.nn.init as init


class ValueHead(nn.Module):
    """ Network head for value determination """
    def __init__(self, input_dim):
        super().__init__()

        self.linear_layer = nn.Linear(input_dim, 1)

    def reset_weights(self):
        # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        init.orthogonal_(self.linear_layer.weight, gain=1.0)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data):
        return self.linear_layer(input_data)[:, 0]
