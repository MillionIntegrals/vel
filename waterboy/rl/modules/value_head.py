import numpy as np
import torch.nn as nn
import torch.nn.init as init


class ValueHead(nn.Module):
    """ Network head for value determination """
    def __init__(self, input_dim):
        super().__init__()

        self.linear_layer = nn.Linear(input_dim, 1)

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
        return self.linear_layer(input_data)[:, 0]
